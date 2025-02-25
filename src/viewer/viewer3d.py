import open3d as o3d

class Viewer3D:
    def __init__(self):
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(window_name="3D Viewer", width=800, height=600)
        
        # 렌더 옵션 설정
        opt = self.vis.get_render_option()
        opt.background_color = [1, 1, 1]  # 흰색 배경
        opt.show_coordinate_frame = True  # 좌표축 표시
        opt.point_size = 1.0
        
        self.setup_camera()
        self.geometries = []
        self.is_visible = False
        
    def setup_camera(self):
        ctr = self.vis.get_view_control()
        ctr.set_zoom(0.8)
        ctr.set_front([0, 0, -1])
        ctr.set_lookat([0, 0, 0])
        ctr.set_up([0, -1, 0])
        
    def add_model(self, model):
        if model and model.mesh:
            self.vis.add_geometry(model.mesh, reset_bounding_box=True)
            self.geometries.append(model.mesh)
            self.update()
            
    def add_geometry(self, geometry):
        """포인트 클라우드나 메쉬 추가"""
        if geometry:
            self.vis.add_geometry(geometry, reset_bounding_box=False)
            self.geometries.append(geometry)
            self.update()
    
    def clear(self):
        """모든 지오메트리 제거"""
        for geometry in self.geometries:
            self.vis.remove_geometry(geometry, reset_bounding_box=False)
        self.geometries.clear()
    
    def update(self):
        """뷰어 업데이트"""
        if self.is_visible:
            for geometry in self.geometries:
                self.vis.update_geometry(geometry)
            self.vis.poll_events()
            self.vis.update_renderer()
    
    def show(self):
        """뷰어 실행"""
        self.is_visible = True
        self.vis.run()
    
    def close(self):
        """뷰어 종료"""
        self.is_visible = False
        self.vis.destroy_window()
        