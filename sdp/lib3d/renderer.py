import os.path
from enum import Enum
from pathlib import Path
from typing import Optional, Dict, List, Any, Tuple

import cv2
import glfw
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R

from OpenGL.GL import *
from OpenGL.GL import shaders
from OpenGL.GLUT import *
from OpenGL.GLU import *

from sdp.lib3d.io import Mesh


class OutputType(Enum):
    Normals = 0
    Noc = 1
    Overlay = 2


Color4i = Tuple[int, int, int, int]
Color4f = Tuple[float, float, float, float]


def color_to_float(col: Color4i) -> Color4f:
    return col[0] / 255, col[1] / 255, col[2] / 255, col[3] / 255


def calc_proj_mat(width: int, height: int, focus: float, n: float = 0.01, f: float = 10.0) -> np.ndarray:
    ratio = n / focus
    l, r = -width / 2 * ratio, width / 2 * ratio
    b, t = -height / 2 * ratio, height / 2 * ratio
    res = np.array([
        [2 * n / (r - l), 0, (r + l) / (r - l), 0],
        [0, 2 * n / (t - b), (t + b) / (t - b), 0],
        [0, 0, -(f + n) / (f - n), -2 * f * n / (f - n)],
        [0, 0, -1, 0],
    ])
    return res.astype(np.float32)


class ProgContainer:
    def __init__(self):
        self.vertex_shader = shaders.compileShader("""
        #version 330 core
        const float M_PI = 3.1415926535897932384626433832795;
        layout(location = 0) in vec3 position;
        layout(location = 1) in vec3 normal;
        uniform mat4 obj_cam_mat;
        uniform mat4 proj_mat;
        uniform int draw_type; // 0 - normals, 1 - noc, 2 - overlay
        uniform float max_dist_to_center;
        uniform vec4 obj_color;

        out vec4 color_out;
        out vec4 pos_out;

        void main()
        {
            pos_out = obj_cam_mat * vec4(position, 1.0);

            gl_Position = proj_mat * pos_out;
            mat3 rot = mat3(obj_cam_mat);
            vec3 col;

            if (draw_type == 0) { // Normals
                vec3 norm_pos = rot * normal;
                col = norm_pos / 2.0 + 0.5;
                color_out = vec4(col, 1);
            } else if (draw_type == 1) { // NOC
                vec3 cam_pos = rot * position;
                col = cam_pos / max_dist_to_center / 2.0 + 0.5;
                //col = normalize(cam_pos) / 2.0 + 0.5;
                color_out = vec4(col, 1);
            } else if (draw_type == 2) { // Overlay
                color_out = obj_color;
            }
        }
        """, GL_VERTEX_SHADER)

        self.geometry_shader = shaders.compileShader("""
        #version 330 core
        layout (triangles) in;
        layout (triangle_strip, max_vertices = 3) out;

        uniform mat4 obj_cam_mat;
        uniform int draw_type;
        in vec4 color_out[3];
        in vec4 pos_out[3];
        out vec4 v_color;

        void main()
        {
            if (draw_type == 0) { // Normals
                vec3 t1 = pos_out[1].xyz - pos_out[0].xyz;        
                vec3 t2 = pos_out[2].xyz - pos_out[0].xyz;
                if (length(t1) < 1e-7 || length(t2) < 1e-7) {
                    v_color = vec4(0.0, 0.0, 0.0, 1.0);
                } else {
                    vec3 n = normalize(cross(t1, t2));
                    vec3 center = (pos_out[0].xyz + pos_out[1].xyz + pos_out[2].xyz) / 3;
                    //vec3 to_center = normalize(obj_cam_mat[3].xyz - center);
                    vec3 to_center = normalize(-center);
                    if (dot(n, to_center) < 0) {
                        n = -n;
                    }
                    // v_color = vec4(n * 0.5 + 0.5, 1.0);
                    float r = min(max(n.x / 2 + 0.5, 0), 1.0);
                    float g = min(max(n.y / 2 + 0.5, 0), 1.0);
                    float b = min(max(n.z / 2 + 0.5, 0), 1.0);
                    v_color = vec4(r, g, b, 1.0);
                }
                for (int i = 0; i < 3; i++) {
                    gl_Position = gl_in[i].gl_Position;
                    EmitVertex();
                }
                EndPrimitive();
            } else if (draw_type == 1 || draw_type == 2 || draw_type == 0) { // NOC or Overlay
                for (int i = 0; i < 3; i++) {
                    gl_Position = gl_in[i].gl_Position;
                    v_color = color_out[i];
                    EmitVertex();
                }
                EndPrimitive();
            }
        }
        """, GL_GEOMETRY_SHADER)

        self.fragment_shader = shaders.compileShader("""
        #version 330 core
        precision highp float;
        in vec4 v_color;
        out vec4 outputColor;
        void main()
        {
            outputColor = v_color;
        }
        """, GL_FRAGMENT_SHADER)
        # self.program = shaders.compileProgram(self.vertex_shader, self.geometry_shader, self.fragment_shader)
        self.program = shaders.compileProgram(self.vertex_shader, self.geometry_shader, self.fragment_shader,
                                              validate=False)
        glUseProgram(self.program)

        self.obj_cam_mat_loc = glGetUniformLocation(self.program, 'obj_cam_mat')
        self.proj_mat_loc = glGetUniformLocation(self.program, 'proj_mat')
        self.draw_type_loc = glGetUniformLocation(self.program, 'draw_type')
        self.obj_color_loc = glGetUniformLocation(self.program, 'obj_color')
        self.max_dist_to_center_loc = glGetUniformLocation(self.program, 'max_dist_to_center')

    def use(self):
        glUseProgram(self.program)


class MeshObj:
    def __init__(self, verts: np.ndarray, normals: np.ndarray, faces: np.ndarray, program: ProgContainer):
        self.program = program
        verts = verts.astype(np.float32)
        norms = normals.astype(np.float32)
        self.verts_norms = np.concatenate([verts, norms], axis=1).astype(np.float32)
        self.faces = faces.astype(np.uint32)
        self.max_dist_to_center = np.max(np.linalg.norm(verts, axis=1))

        self.program.use()

        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)

        self.buffers = glGenBuffers(2)
        glBindBuffer(GL_ARRAY_BUFFER, self.buffers[0])
        glBufferData(GL_ARRAY_BUFFER, self.verts_norms.size * 4, self.verts_norms, GL_STATIC_DRAW)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.buffers[1])
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, self.faces, GL_STATIC_DRAW)

    def draw(self, proj_mat: np.ndarray, obj_cam_mat: np.ndarray, out_type: OutputType,
             obj_color: Optional[Color4i] = None):
        self.program.use()

        # proj_mat = glGetFloat(GL_PROJECTION_MATRIX)
        glUniformMatrix4fv(self.program.obj_cam_mat_loc, 1, True, obj_cam_mat.astype(np.float32))
        glUniformMatrix4fv(self.program.proj_mat_loc, 1, True, proj_mat.astype(np.float32))
        glUniform1i(self.program.draw_type_loc, out_type.value)
        glUniform1f(self.program.max_dist_to_center_loc, self.max_dist_to_center)
        if out_type == OutputType.Overlay and obj_color is not None:
            glUniform4f(self.program.obj_color_loc, *color_to_float(obj_color))

        glBindVertexArray(self.vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.buffers[0])
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.buffers[1])
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * 4, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * 4, ctypes.c_void_p(3 * 4))
        glEnableVertexAttribArray(1)
        glDrawElements(GL_TRIANGLES, self.faces.size, GL_UNSIGNED_INT, None)


class Renderer:
    width: int
    height: int
    title: str
    hide_window: bool
    proj_mat: np.ndarray
    cam_mat: np.ndarray
    window: Any
    prog: ProgContainer
    mesh_objs: Dict[Any, MeshObj]
    cv_to_opengl_mat: np.ndarray

    def __init__(self, meshes: Dict[Any, Mesh], win_size: Tuple[int, int] = (640, 480), title: str = 'Renderer',
                 hide_window: bool = False):
        self.width, self.height = win_size
        self.title = title
        self.hide_window = hide_window
        self.proj_mat = calc_proj_mat(self.width, self.height, self.width)
        self.cam_mat = np.array([
            self.width, 0, self.width / 2,
            0, self.width, self.height / 2,
            0, 0, 1.,
        ]).reshape((3, 3))

        if not glfw.init():
            raise Exception(f'Error. Cannot init GLFW!')

        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 1)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, GL_TRUE)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

        self.window = glfw.create_window(self.width, self.height, self.title, None, None)
        if not self.window:
            raise Exception(f'Error. Cannot create window of size: {self.width}x{self.height}')
        glfw.make_context_current(self.window)
        if self.hide_window:
            glfw.hide_window(self.window)
        self._update_viewport_size()
        glClearColor(0, 0, 0, 0)
        glEnable(GL_DEPTH_TEST)
        glfw.set_window_size_callback(self.window, self._on_window_resize)
        glfw.set_framebuffer_size_callback(self.window, self._on_framebuffer_resize)
        glfw.swap_buffers(self.window)

        self.prog = ProgContainer()

        self.mesh_objs = self.create_mesh_objs(meshes, self.prog)
        self.cv_to_opengl_mat = np.eye(4, dtype=np.float32)
        self.cv_to_opengl_mat[:3, :3] = R.from_rotvec(np.pi * np.array([1, 0, 0])).as_matrix()

    def _update_viewport_size(self):
        width, height = glfw.get_framebuffer_size(self.window)
        glViewport(0, 0, width, height)

    def _on_window_resize(self, win, width, height):
        print(f'_on_window_resize: {width}x{height}')

    def _on_framebuffer_resize(self, win, width, height):
        print(f'_on_framebuffer_resize: {width}x{height}')
        self._update_viewport_size()

    @staticmethod
    def create_mesh_objs(meshes: Dict[Any, Mesh], prog: ProgContainer) -> Dict[Any, MeshObj]:
        return {mid: MeshObj(mesh.vertices, mesh.vertex_normals, mesh.triangles, prog) for mid, mesh in meshes.items()}

    def set_window_size(self, win_size: Tuple[int, int]):
        width, height = win_size
        if (self.width, self.height) == (width, height):
            return
        self.width, self.height = width, height
        self.proj_mat = calc_proj_mat(self.width, self.height, self.cam_mat[0, 0])
        glfw.set_window_size(self.window, self.width, self.height)
        glViewport(0, 0, self.width, self.height)
        for _ in range(2):
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            glfw.swap_buffers(self.window)

    def set_camera_matrix(self, cam_mat: np.ndarray):
        if self.cam_mat is not None and np.allclose(self.cam_mat, cam_mat):
            return
        self.cam_mat = cam_mat.astype(np.float32)
        self.proj_mat = calc_proj_mat(self.width, self.height, self.cam_mat[0, 0])
        # fovy_half_tan = (self.height / 2) / self.cam_mat[1, 1]
        # fovy = np.arctan(fovy_half_tan) * 2 * (180 / np.pi)

        # print(f'Fov y: {fovy:.2f}')
        # glMatrixMode(GL_PROJECTION)
        # glLoadIdentity()
        # gluPerspective(fovy, self.width / self.height, 0.01, 10)
        # glMatrixMode(GL_MODELVIEW)
        # glLoadIdentity()

    def gen_colors(self, cam_mat: np.ndarray, objs_poses: List[Tuple[Any, np.ndarray]], out_type: OutputType,
                   obj_color: Optional[Color4i] = None) -> np.ndarray:
        self.set_camera_matrix(cam_mat)

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        for obj_id, obj_cam_mat in objs_poses:
            mesh_obj = self.mesh_objs[obj_id]
            obj_cam_mat = self.cv_to_opengl_mat @ obj_cam_mat

            mesh_obj.draw(self.proj_mat, obj_cam_mat, out_type, obj_color)

        image_type = GL_RGBA if out_type == OutputType.Overlay else GL_RGB
        width, height = glfw.get_framebuffer_size(self.window)
        image_buffer = glReadPixels(0, 0, width, height, image_type, GL_UNSIGNED_BYTE)
        image = np.frombuffer(image_buffer, dtype=np.uint8).reshape((height, width, -1))
        if width != self.width or height != self.height:
            image = cv2.resize(image, (self.width, self.height), interpolation=cv2.INTER_AREA)
        image = cv2.flip(image, 0)

        glfw.swap_buffers(self.window)

        return image


def gen_rot_vec() -> Tuple[np.ndarray, float]:
    r = np.random.random(2)
    ang1 = 2 * np.pi * r[0]
    cos1, sin1 = np.cos(ang1), np.sin(ang1)
    cos2 = 1 - 2 * r[1]
    sin2 = np.sqrt(1 - cos2**2)

    rvec = np.array((cos1 * sin2, sin1 * sin2, cos2))
    ang = np.random.uniform(0, 2 * np.pi)

    return rvec, ang


def test_renderer():
    data_path = Path(os.path.expandvars('$HOME/data'))
    # data_path = Path(os.path.expandvars('/ws/data'))
    obj_path = data_path / 'sds_data/objs/teamug.stl'
    # obj_path = data_path / 'sds/itodd/models/obj_000004.ply'
    print(f'Loading {obj_path}')
    mesh = Mesh.read_mesh(obj_path)
    mesh_id = 'teamug'
    meshes = {mesh_id: mesh}

    w, h = 640, 480
    w, h = 2048, 2048
    cam_mat = np.array([
        w / 2, 0., w / 2,
        0, w / 2, h / 2,
        0, 0, 1,
    ]).reshape((3, 3))

    ren = Renderer(meshes, (w, h), hide_window=True)
    ren.set_camera_matrix(cam_mat)

    np.random.seed(1)
    pos_center = np.array((0., 0., 1.))
    while True:
        objs = []

        for i in range(25):
            rot_vec, angle = gen_rot_vec()
            rot = R.from_rotvec(rot_vec * angle).as_matrix()
            pos = pos_center + np.random.uniform((-.6, -.6, -0.2), (.6, .6, 1))

            H = np.eye(4)
            H[:3, :3] = rot
            H[:3, 3] = pos
            objs.append((mesh_id, H))

        img_noc = ren.gen_colors(cam_mat, objs, OutputType.Noc)
        img_normals = ren.gen_colors(cam_mat, objs, OutputType.Normals)

        img_noc = cv2.cvtColor(img_noc, cv2.COLOR_RGB2BGR)
        img_normals = cv2.cvtColor(img_normals, cv2.COLOR_RGB2BGR)

        cv2.imshow('noc', img_noc)
        cv2.imshow('normals', img_normals)

        if cv2.waitKey() in (27, ord('q')):
            break


if __name__ == '__main__':
    test_renderer()

