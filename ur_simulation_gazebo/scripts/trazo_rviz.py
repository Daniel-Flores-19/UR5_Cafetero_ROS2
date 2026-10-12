#!/usr/bin/env python3
"""
Dibuja en RViz el trazo del plumón.

Sigue tool0 por TF (base_link -> tool0) y, mientras la brida está a menos de
`umbral` sobre z_papel, considera que el plumón toca el papel y añade el punto
al trazo actual. Al levantarse cierra el trazo; el siguiente contacto abre uno
nuevo. Publica un MarkerArray (un LINE_STRIP por trazo) en /pintor/trazo.

z_papel es la Z de la brida con el plumón en el papel (la misma dist_z que usa
move_draw_sub), en el marco base del robot.
"""
import rclpy
from geometry_msgs.msg import Point
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.time import Time
from tf2_ros import Buffer, TransformListener
from visualization_msgs.msg import Marker, MarkerArray


class TrazoRviz(Node):
    def __init__(self):
        super().__init__('trazo_rviz')
        self.declare_parameter('z_papel', 0.1918)
        self.declare_parameter('umbral', 0.0015)       # [m] sobre z_papel (el levantamiento es ~11 mm)
        self.declare_parameter('frame', 'base_link')
        self.declare_parameter('efector', 'tool0')
        self.declare_parameter('paso_min', 0.0005)     # [m] entre puntos guardados
        self.declare_parameter('grosor', 0.0015)       # [m] ancho de la línea

        self.z_papel = self.get_parameter('z_papel').value
        self.umbral = self.get_parameter('umbral').value
        self.frame = self.get_parameter('frame').value
        self.efector = self.get_parameter('efector').value
        self.paso_min = self.get_parameter('paso_min').value
        self.grosor = self.get_parameter('grosor').value

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.pub = self.create_publisher(MarkerArray, '/pintor/trazo', 10)

        self.trazos = []        # lista de listas de Point
        self.abajo = False

        self.create_timer(0.02, self.muestrear)   # 50 Hz
        self.create_timer(0.2, self.publicar)     # 5 Hz
        self.get_logger().info(
            f'Trazo en /pintor/trazo: plumón abajo si Z({self.efector}) < '
            f'{self.z_papel + self.umbral:.4f} m en {self.frame}')

    def muestrear(self):
        try:
            t = self.tf_buffer.lookup_transform(
                self.frame, self.efector, Time(), timeout=Duration(seconds=0.0))
        except Exception:
            return
        p = t.transform.translation
        abajo = p.z < self.z_papel + self.umbral

        if abajo and not self.abajo:
            self.trazos.append([])
        self.abajo = abajo
        if not abajo:
            return

        trazo = self.trazos[-1]
        if trazo:
            u = trazo[-1]
            if (u.x - p.x) ** 2 + (u.y - p.y) ** 2 < self.paso_min ** 2:
                return
        # Se dibuja sobre el plano del papel, no a la altura real de la brida.
        trazo.append(Point(x=p.x, y=p.y, z=self.z_papel))

    def publicar(self):
        msg = MarkerArray()
        ahora = self.get_clock().now().to_msg()
        for i, trazo in enumerate(self.trazos):
            if len(trazo) < 2:
                continue
            m = Marker()
            m.header.frame_id = self.frame
            m.header.stamp = ahora
            m.ns = 'trazo'
            m.id = i
            m.type = Marker.LINE_STRIP
            m.action = Marker.ADD
            m.pose.orientation.w = 1.0
            m.scale.x = self.grosor
            m.color.r, m.color.g, m.color.b, m.color.a = 0.05, 0.1, 0.8, 1.0
            m.points = trazo
            msg.markers.append(m)
        if msg.markers:
            self.pub.publish(msg)


def main():
    rclpy.init()
    nodo = TrazoRviz()
    try:
        rclpy.spin(nodo)
    except KeyboardInterrupt:
        pass
    finally:
        nodo.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
