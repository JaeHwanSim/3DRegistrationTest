import open3d as o3d

class STLModel:
    def __init__(self):
        self.mesh = None
        self.color = None
        self.name = None
        
    def load(self, file_path):
        try:
            self.mesh = o3d.io.read_triangle_mesh(file_path)
            self.name = file_path.split("/")[-1]
            self.mesh.compute_vertex_normals()
            self.set_color([1, 0.7, 0])
            return True
        except Exception as e:
            print(f"STL 모델 로드 오류: {e}")
            return False
    
    def set_color(self, color):
        self.color = color
        self.mesh.paint_uniform_color(color)

    def get_mesh(self):
        return self.mesh
    
    def get_color(self):
        return self.color
    
    def get_name(self):
        return self.name
    
    def transform(self, transformation_matrix):
        """메쉬에 변환 행렬 적용"""
        if self.mesh:
            # Open3D의 변환 메서드 사용
            self.mesh = self.mesh.transform(transformation_matrix)
            return True
        return False


