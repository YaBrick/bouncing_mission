#!/usr/bin/env python3
import rospy
import cv2
import numpy as np
import threading
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped, Point
from ultralytics import YOLO

class VisaoMissaoCompeticao:
    def __init__(self):
        rospy.init_node('visao_missao_node', anonymous=True)
        self.bridge = CvBridge()

        # Estado do Drone
        self.drone_x, self.drone_y, self.drone_z = 0.0, 0.0, 0.0
        self.posicao_recebida = False
        self.comando_pouso_enviado = False

        # Memória da Missão
        self.gabarito_encontrado = False
        self.gabarito_forma = None
        self.gabarito_id = None
        self.bases_encontradas = []

        # Memória de qual base foi autorizada para rastreio contínuo
        self.alvo_nome_confirmado = None

        # Controle de Frames
        self.latest_frame = None
        self.latest_header = None
        self.frame_lock = threading.Lock()
        self.rate = rospy.Rate(5)

        self._configurar_modelos()

        # ROS Subscribers e Publishers
        rospy.Subscriber("/mavros/local_position/pose", PoseStamped, self.pos_callback)
        rospy.Subscriber("/camera/image_raw", Image, self.image_callback, queue_size=1, buff_size=2**24)

        self.image_pub = rospy.Publisher("/visao/resultado_yolo_aruco", Image, queue_size=1)
        self.alvo_pub = rospy.Publisher("/visao/alvo_pouso", Point, queue_size=10)
        self.erro_pub = rospy.Publisher("/visao/erro_alvo", Point, queue_size=10)

        rospy.loginfo("Nó de Visão da Missão Boucing 2.0 Iniciado! (Rodando a 5 FPS)")

    def _configurar_modelos(self):
        try:
            self.model = YOLO("best.pt")
            rospy.loginfo("Modelo YOLO carregado com sucesso!")
        except Exception as e:
            rospy.logerr(f"Erro ao carregar YOLO: {e}")
            self.model = None

        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_1000)
        self.aruco_parametros = cv2.aruco.DetectorParameters()
        self.aruco_detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_parametros)

    def pos_callback(self, msg):
        self.drone_x = msg.pose.position.x
        self.drone_y = msg.pose.position.y
        self.drone_z = msg.pose.position.z
        self.posicao_recebida = True

    def image_callback(self, data):
        try:
            frame = self.bridge.imgmsg_to_cv2(data, "bgr8")
            with self.frame_lock:
                self.latest_frame = frame
                self.latest_header = data.header
        except CvBridgeError as e:
            rospy.logerr(e)

    def projected_distance(self, u, v, h):
        """Converte a posição em pixels para metros relativos baseados na câmera do Gazebo"""
        fx = 155.1
        fy = 147.3
        cx = 820.0
        cy = 616.0

        h = float(max(0.1, h))
        X = (float(u) - cx) * h / fx
        Y = -(float(v) - cy) * h / fy

        return X, Y

    def enviar_comando_pouso(self, x, y):
        if self.comando_pouso_enviado:
            return

        msg_alvo = Point(x=x, y=y, z=0.0)
        self.alvo_pub.publish(msg_alvo)
        self.comando_pouso_enviado = True
        rospy.logwarn(f"COMANDO DE POUSO ENVIADO AO MAVROS PARA X:{x:.2f}, Y:{y:.2f}")

    def _avaliar_condicao_pouso(self, forma_base, numero_base, x, y, nome_base):
        if not self.gabarito_encontrado:
            return False

        if (forma_base == self.gabarito_forma) and (self.gabarito_id % numero_base == 0):
            rospy.loginfo("="*50)
            rospy.loginfo(f"POUSO AUTORIZADO! BASE CORRETA: {nome_base.upper()}")
            rospy.loginfo(f"   Vá para as Coordenadas: X:{x:.2f}, Y:{y:.2f}")
            rospy.loginfo("="*50)

            # Grava o nome do alvo para começar o rastreio contínuo
            self.alvo_nome_confirmado = nome_base
            self.enviar_comando_pouso(x, y)
            return True
        return False

    def _processar_novo_gabarito(self, forma, aruco_id):
        if self.gabarito_encontrado:
            return

        self.gabarito_encontrado = True
        self.gabarito_forma = forma
        self.gabarito_id = aruco_id

        rospy.loginfo("*"*50)
        rospy.loginfo(f"GABARITO MEMORIZADO! Forma: {forma.upper()} | ID: {aruco_id}")
        rospy.loginfo("*"*50)

        for base in self.bases_encontradas:
            self._avaliar_condicao_pouso(base['forma'], base['numero'], base['x'], base['y'], base['nome'])

    def _processar_nova_base(self, nome, forma, numero):
        if any(b['nome'] == nome for b in self.bases_encontradas):
            return

        nova_base = {'nome': nome, 'forma': forma, 'numero': numero, 'x': self.drone_x, 'y': self.drone_y}
        self.bases_encontradas.append(nova_base)
        rospy.loginfo(f"BASE MAPEADA: {nome.upper()} em X:{self.drone_x:.2f}, Y:{self.drone_y:.2f}")

        self._avaliar_condicao_pouso(forma, numero, self.drone_x, self.drone_y, nome)

    def _fazer_fusao_dados(self, deteccoes_yolo, deteccoes_aruco):
        for yolo in deteccoes_yolo:
            nome_yolo = yolo['nome']

            # Se o YOLO garante que é uma classe de Gabarito (termina com 'aruco')
            if nome_yolo.endswith('aruco'):
                # Extrai apenas o nome da forma (ex: 'hexagonoaruco' -> 'hexagono')
                forma = nome_yolo.replace('aruco', '')
                xmin, ymin, xmax, ymax = yolo['bbox']

                for aruco in deteccoes_aruco:
                    # Verifica se o ArUco está dentro da bounding box detectada pelo YOLO
                    if xmin <= aruco['cx'] <= xmax and ymin <= aruco['cy'] <= ymax:
                        yolo['eh_gabarito'] = True
                        self._processar_novo_gabarito(forma, aruco['id'])
                        break

            # Se for uma base comum
            else:
                forma = ''.join([letra for letra in nome_yolo if not letra.isdigit()])
                numero_str = ''.join([num for num in nome_yolo if num.isdigit()])
                numero = int(numero_str) if numero_str else None

                # Mapeia como base desde que não tenha sido flaggada acidentalmente
                if numero is not None and not yolo.get('eh_gabarito', False):
                    self._processar_nova_base(nome_yolo, forma, numero)

    def rodar(self):
        while not rospy.is_shutdown():
            with self.frame_lock:
                if self.latest_frame is None:
                    self.rate.sleep()
                    continue
                frame = self.latest_frame.copy()
                header_atual = self.latest_header
                self.latest_frame = None

            frame_anotado = frame.copy()
            deteccoes_yolo = []
            deteccoes_aruco = []

            # Processamento YOLO
            if self.model is not None:
                resultados = self.model.predict(source=frame, conf=0.3, verbose=False, device='cpu')
                frame_anotado = resultados[0].plot()

                for caixa in resultados[0].boxes:
                    nome = self.model.names[int(caixa.cls[0])]
                    coords = caixa.xyxy[0].tolist()
                    xmin, ymin, xmax, ymax = coords
                    cx = int((xmin + xmax) / 2)
                    cy = int((ymin + ymax) / 2)
                    
                    deteccoes_yolo.append({
                        'nome': nome, 
                        'cx': cx, 
                        'cy': cy, 
                        'bbox': coords, 
                        'eh_gabarito': False
                    })

            # Processamento ArUco
            corners, ids, _ = self.aruco_detector.detectMarkers(frame)

            if ids is not None:
                cv2.aruco.drawDetectedMarkers(frame_anotado, corners, ids)
                for i, cantos in enumerate(corners):
                    cx, cy = np.mean(cantos[0], axis=0).astype(int)
                    deteccoes_aruco.append({'id': int(ids[i][0]), 'cx': cx, 'cy': cy})

            # Fusão e Tomada de Decisão
            self._fazer_fusao_dados(deteccoes_yolo, deteccoes_aruco)

            # Rastreio Contínuo da Base Alvo para Pouso Ativo
            if self.alvo_nome_confirmado is not None:
                for yolo in deteccoes_yolo:
                    
                    # Ignora a figura se for gabarito ou se pertencer às classes com ArUco
                    if yolo.get('eh_gabarito', False) or yolo['nome'].endswith('aruco'):
                        continue

                    if yolo['nome'] == self.alvo_nome_confirmado:
                        dist_x, dist_y = self.projected_distance(yolo['cx'], yolo['cy'], self.drone_z)

                        msg_erro = Point()
                        msg_erro.x = -dist_y
                        msg_erro.y = dist_x
                        msg_erro.z = 0.0
                        self.erro_pub.publish(msg_erro)

                        # Desenha círculo verde para confirmar o travamento do alvo no frame
                        cv2.circle(frame_anotado, (yolo['cx'], yolo['cy']), 10, (0, 255, 0), -1)
                        break

            # Publica a imagem anotada
            try:
                msg_publicar = self.bridge.cv2_to_imgmsg(frame_anotado, "bgr8")
                msg_publicar.header = header_atual
                self.image_pub.publish(msg_publicar)
            except CvBridgeError as e:
                rospy.logerr(e)

            self.rate.sleep()

if __name__ == '__main__':
    visao = VisaoMissaoCompeticao()
    try:
        visao.rodar()
    except rospy.ROSInterruptException:
        pass
