#!/usr/bin/env python3
import rospy
import math
from geometry_msgs.msg import PoseStamped, TwistStamped, Point
from mavros_msgs.msg import State
from mavros_msgs.srv import CommandBool, SetMode, CommandTOL

class ZigZagVelocidade:
    def __init__(self):
        rospy.init_node('zigzag_vel_node', anonymous=True)

        self.estado = State()
        self.pos = PoseStamped()

        # Variáveis de Pouso
        self.pouso_acionado = False
        self.alvo_x = 0.0
        self.alvo_y = 0.0

        # Variáveis de Correção Visual
        self.erro_x = 0.0
        self.erro_y = 0.0
        self.ultima_visao_tempo = rospy.Time(0)

        # Subscribers
        rospy.Subscriber("mavros/state", State, self.state_cb)
        rospy.Subscriber("mavros/local_position/pose", PoseStamped, self.pos_cb)
        rospy.Subscriber("/visao/alvo_pouso", Point, self.alvo_cb)
        rospy.Subscriber("/visao/erro_alvo", Point, self.erro_visao_cb)

        # Publisher de velocidade
        self.pub_vel = rospy.Publisher("mavros/setpoint_velocity/cmd_vel", TwistStamped, queue_size=10)

        # Serviços
        rospy.wait_for_service('mavros/cmd/arming')
        rospy.wait_for_service('mavros/set_mode')
        rospy.wait_for_service('mavros/cmd/takeoff')
        rospy.wait_for_service('mavros/cmd/land')

        self.arm = rospy.ServiceProxy('mavros/cmd/arming', CommandBool)
        self.set_mode = rospy.ServiceProxy('mavros/set_mode', SetMode)
        self.takeoff = rospy.ServiceProxy('mavros/cmd/takeoff', CommandTOL)
        self.land = rospy.ServiceProxy('mavros/cmd/land', CommandTOL)
        self.rate = rospy.Rate(10)

    def state_cb(self, msg):
        self.estado = msg

    def pos_cb(self, msg):
        self.pos = msg

    def alvo_cb(self, msg):
        if not self.pouso_acionado:
            self.alvo_x = msg.x
            self.alvo_y = msg.y
            self.pouso_acionado = True
            rospy.logwarn(f"ALERTA GERAL: Alvo recebido (X:{self.alvo_x:.2f}, Y:{self.alvo_y:.2f}). Abortando busca!")

    def erro_visao_cb(self, msg):
        self.erro_x = msg.x
        self.erro_y = msg.y
        self.ultima_visao_tempo = rospy.Time.now()

    def esperar_conexao(self):
        rospy.loginfo("Aguardando conexão...")
        while not rospy.is_shutdown() and not self.estado.connected:
            self.rate.sleep()
        rospy.loginfo("Conectado!")

    def armar_e_decolar(self, alt):
        vel = TwistStamped()
        for _ in range(50):
            self.pub_vel.publish(vel)
            self.rate.sleep()

        while self.estado.mode != "GUIDED":
            self.set_mode(custom_mode="GUIDED")
            self.rate.sleep()

        while not self.estado.armed:
            self.arm(True)
            self.rate.sleep()

        rospy.loginfo("Decolando...")
        self.takeoff(altitude=alt)
        while self.pos.pose.position.z < alt * 0.95:
            self.rate.sleep()

        rospy.loginfo("Altitude atingida")

    def mover_para(self, x_alvo, y_alvo, z_alvo, vel=0.5, tol=0.3, ignorar_interrupcao=False):
        while not rospy.is_shutdown():
            if self.pouso_acionado and not ignorar_interrupcao:
                return

            x = self.pos.pose.position.x
            y = self.pos.pose.position.y
            z = self.pos.pose.position.z

            dx = x_alvo - x
            dy = y_alvo - y
            dz = z_alvo - z

            dist = math.sqrt(dx**2 + dy**2)

            if dist < tol:
                break

            vel_msg = TwistStamped()
            vel_msg.twist.linear.x = vel * (dx / dist)
            vel_msg.twist.linear.y = vel * (dy / dist)
            vel_msg.twist.linear.z = dz * 0.5

            self.pub_vel.publish(vel_msg)
            self.rate.sleep()

        vel_msg = TwistStamped()
        self.pub_vel.publish(vel_msg)

    def executar_zigzag(self, tam_x, tam_y, passo, alt):
        x_min, x_max = -tam_x/2, tam_x/2
        y_min, y_max = -tam_y/2, tam_y/2

        y = y_min
        direcao = 1
        
        rospy.loginfo("Iniciando zig-zag...")
        self.mover_para(x_min, y_min, alt)

        while y <= y_max and not rospy.is_shutdown():
            if self.pouso_acionado:
                rospy.logwarn("Zig-zag interrompido pela Visão!")
                return

            x_dest = x_max if direcao == 1 else x_min
            self.mover_para(x_dest, y, alt)

            y += passo
            direcao *= -1

            if y <= y_max:
                self.mover_para(x_dest, y, alt)

        rospy.loginfo("Zig-zag finalizado")

    def corrigir_e_pousar(self):
        rospy.loginfo("================================================")
        rospy.loginfo(" Pousando com correção visual ativa ")
        rospy.loginfo("================================================")
        kp = 0.9

        while not rospy.is_shutdown():
            tempo_sem_ver = (rospy.Time.now() - self.ultima_visao_tempo).to_sec()
            vel_msg = TwistStamped()

            if tempo_sem_ver < 1.5:
                # Se estamos vendo a base, corrige XY e ajusta a descida Z
                vel_msg.twist.linear.x = kp * self.erro_x
                vel_msg.twist.linear.y = kp * self.erro_y
                
                # Se estiver bem em cima do centro (erro < 0.3m), desce mais rápido
                erro_total = math.sqrt(self.erro_x**2 + self.erro_y**2)
                if erro_total < 0.15:
                    vel_msg.twist.linear.z = -0.3 # Desce confiantemente
                else:
                    vel_msg.twist.linear.z = -0.05 # Desce quase parando enquanto corrige XY
            else:
                # Perdeu de vista. Para de corrigir XY e desce cego na esperança de ver de novo
                rospy.logwarn_throttle(2, "Alvo perdido de vista! Descendo cautelosamente...")
                vel_msg.twist.linear.x = 0.0
                vel_msg.twist.linear.y = 0.0
                vel_msg.twist.linear.z = -0.15

            self.pub_vel.publish(vel_msg)

            # Quando estiver muito perto do chão (ex: 30cm), manda o comando definitivo
            if self.pos.pose.position.z < 0.3:
                rospy.logwarn("Proximidade do solo! Acionando LAND final do MAVROS.")
                self.set_mode(custom_mode="LAND")
                break

            self.rate.sleep()


if __name__ == "__main__":
    try:
        drone = ZigZagVelocidade()
        drone.esperar_conexao()

        ALT = 1.5

        drone.armar_e_decolar(ALT)
        rospy.sleep(2)

        while not rospy.is_shutdown():
            if not drone.pouso_acionado:
                drone.executar_zigzag(10, 4, 1.0, ALT)
            else:
                rospy.loginfo(f"Navegando para as imediações da base (X:{drone.alvo_x:.2f}, Y:{drone.alvo_y:.2f})")

                # 1. Volta para as coordenadas onde viu a base pela primeira vez
                drone.mover_para(drone.alvo_x, drone.alvo_y, ALT, ignorar_interrupcao=True)

                # 2. Chama o procedimento de descida ativa por câmera
                drone.corrigir_e_pousar()

                rospy.loginfo("Missão Concluída! Desligando nó de navegação.")
                break

    except rospy.ROSInterruptException:
        pass
