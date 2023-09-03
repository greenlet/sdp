from pathlib import Path
from typing import Optional, Any, Union

import numpy as np
import open3d as o3d


class Mesh:
    mesh_o3d: o3d.geometry.TriangleMesh
    _vertices: Optional[np.ndarray] = None
    _triangles: Optional[np.ndarray] = None
    _triangle_normals: Optional[np.ndarray] = None
    _vertex_normals: Optional[np.ndarray] = None

    def __init__(self, mesh_o3d: o3d.geometry.TriangleMesh):
        self.mesh_o3d = mesh_o3d

    @property
    def vertices(self) -> np.ndarray:
        if self._vertices is None:
            self._vertices = np.asarray(self.mesh_o3d.vertices)
        return self._vertices

    @property
    def triangles(self) -> np.ndarray:
        if self._triangles is None:
            self._triangles = np.asarray(self.mesh_o3d.triangles)
        return self._triangles

    @property
    def vertex_normals(self) -> np.ndarray:
        if self._vertex_normals is None:
            self.mesh_o3d.compute_vertex_normals()
            self._vertex_normals = np.asarray(self.mesh_o3d.vertex_normals)
        return self._vertex_normals

    @property
    def triangle_normals(self) -> np.ndarray:
        if self._triangle_normals is None:
            self.mesh_o3d.compute_triangle_normals()
            self._triangle_normals = np.asarray(self.mesh_o3d.triangle_normals)
        return self._triangle_normals

    @staticmethod
    def read_mesh(fpath: Path) -> 'Mesh':
        mesh_o3d = o3d.io.read_triangle_mesh(str(fpath))
        return Mesh(mesh_o3d)


