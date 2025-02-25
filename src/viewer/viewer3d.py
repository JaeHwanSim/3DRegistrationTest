import open3d as o3d

class Viewer3D:
    def __init__(self):
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(window_name="3D Viewer", width=800, height=600)
        # 기본 카메라 뷰 설정
        self.setup_camera()
        self.geometries = []  # 현재 표시된 모델들 추적
        
    def setup_camera(self):
        ctr = self.vis.get_view_control()
        ctr.set_zoom(0.8)
        ctr.set_front([0, 0, -1])
        ctr.set_lookat([0, 0, 0])
        ctr.set_up([0, -1, 0])
        
    def clear(self):
        """모든 지오메트리 제거"""
        for geometry in self.geometries:
            self.vis.remove_geometry(geometry, reset_bounding_box=False)
        self.geometries.clear()
    
    def add_model(self, model):
        if model and model.mesh:
            self.vis.add_geometry(model.mesh, reset_bounding_box=True)
            self.geometries.append(model.mesh)
    
    def update(self):
        """뷰어 업데이트"""
        self.vis.poll_events()
        self.vis.update_renderer()
    
    def show(self):
        self.vis.run()
    
    def close(self):
        self.vis.destroy_window()
        