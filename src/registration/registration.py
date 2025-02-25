import open3d as o3d
import numpy as np
from scipy.spatial import KDTree

class Registration:
    def __init__(self):
        self.source = None
        self.target = None
        self.transformation = np.identity(4)
        
    def mesh_to_pointcloud(self, mesh):
        """TriangleMesh를 PointCloud로 변환"""
        pcd = o3d.geometry.PointCloud()
        pcd.points = mesh.vertices
        pcd.colors = mesh.vertex_colors
        pcd.normals = mesh.vertex_normals
        return pcd

    def align_to_principal_axes(self, pcd):
        """주축 기준 정렬"""
        # 중심점 계산
        center = np.mean(np.asarray(pcd.points), axis=0)
        
        # 중심으로 이동
        points_centered = np.asarray(pcd.points) - center
        
        # PCA로 주축 계산
        covariance = np.cov(points_centered.T)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        
        # y축이 위를 향하도록 정렬
        R = eigenvectors.T
        if R[1, 1] < 0:  # y축 방향 확인
            R[1] = -R[1]
        
        # 변환 행렬 생성
        transformation = np.identity(4)
        transformation[:3, :3] = R
        transformation[:3, 3] = -R @ center
        
        return transformation
        
    def preprocess_point_cloud(self, pcd, voxel_size=0.08):
        """포인트 클라우드 전처리"""
        # 주축 기준 정렬
        init_transform = self.align_to_principal_axes(pcd)
        pcd.transform(init_transform)
        
        # 다운샘플링
        pcd_down = pcd.voxel_down_sample(voxel_size)
        
        # 법선 벡터 계산
        pcd_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
        
        # FPFH 특징점 계산
        pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            pcd_down,
            o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5, max_nn=100))
        
        return pcd_down, pcd_fpfh, init_transform

    def execute_global_registration(self, source_down, target_down, 
                                  source_fpfh, target_fpfh, voxel_size):
        """전역 정합"""
        distance_threshold = voxel_size * 1.5
        
        result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            source_down, target_down, source_fpfh, target_fpfh,
            True,
            distance_threshold,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
            4,  # 최소 대응점 수 증가
            [
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
            ],
            o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 100))
        return result

    def refine_registration(self, source, target, transformation, voxel_size):
        """다단계 ICP 정합"""
        current_transformation = transformation
        
        # 두 단계의 ICP 실행
        for distance_multiplier in [1.0, 0.5]:
            distance_threshold = voxel_size * distance_multiplier
            
            result = o3d.pipelines.registration.registration_icp(
                source, target,
                distance_threshold,
                current_transformation,
                o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                o3d.pipelines.registration.ICPConvergenceCriteria(
                    relative_fitness=1e-6,
                    relative_rmse=1e-6,
                    max_iteration=50))
            
            current_transformation = result.transformation
        
        return result

    def register(self, source_model, target_model):
        """정합 프로세스"""
        # 메쉬를 포인트 클라우드로 변환
        source_pcd = self.mesh_to_pointcloud(source_model.mesh)
        target_pcd = self.mesh_to_pointcloud(target_model.mesh)
        
        # 전처리 및 초기 정렬
        voxel_size = 0.08
        source_down, source_fpfh, source_init = self.preprocess_point_cloud(source_pcd, voxel_size)
        target_down, target_fpfh, target_init = self.preprocess_point_cloud(target_pcd, voxel_size)
        
        print(f"다운샘플링 후 포인트 수: source={len(source_down.points)}, target={len(target_down.points)}")
        
        # 전역 정합
        result_global = self.execute_global_registration(
            source_down, target_down, source_fpfh, target_fpfh, voxel_size)
        
        # 정밀 정합
        result_icp = self.refine_registration(
            source_down, target_down,  # 다운샘플링된 데이터 사용
            result_global.transformation, voxel_size)
        
        # 최종 변환 행렬 계산 (초기 정렬 + 전역 정합 + 정밀 정합)
        self.transformation = np.dot(result_icp.transformation, 
                                   np.dot(result_global.transformation, 
                                         source_init))
        
        # 타겟 모델을 원래 위치로 되돌림
        target_model.mesh.transform(np.linalg.inv(target_init))
        
        # 소스 메쉬에 최종 변환 적용
        source_model.mesh.transform(self.transformation)
        
        return {
            'fitness': result_icp.fitness,
            'inlier_rmse': result_icp.inlier_rmse,
            'transformation': self.transformation,
            'global_fitness': result_global.fitness,
            'global_rmse': result_global.inlier_rmse
        }

    def execute_pca_alignment(self, source_model, target_model):
        """PCA 기반 주축 정렬"""
        # 포인트 클라우드 변환
        source_pcd = self.mesh_to_pointcloud(source_model.mesh)
        target_pcd = self.mesh_to_pointcloud(target_model.mesh)
        
        # 소스 모델 중심점과 주축 계산
        source_points = np.asarray(source_pcd.points)
        source_mean = np.mean(source_points, axis=0)
        source_covariance = np.cov(source_points.T)
        source_eigenvalues, source_eigenvectors = np.linalg.eigh(source_covariance)
        
        # 타겟 모델 중심점과 주축 계산
        target_points = np.asarray(target_pcd.points)
        target_mean = np.mean(target_points, axis=0)
        target_covariance = np.cov(target_points.T)
        target_eigenvalues, target_eigenvectors = np.linalg.eigh(target_covariance)
        
        # 회전 행렬 계산
        R = np.dot(target_eigenvectors, source_eigenvectors.T)
        
        # y축이 위를 향하도록 조정
        if R[1, 1] < 0:
            R[:, 1] = -R[:, 1]
        
        # 변환 행렬 생성
        transformation = np.identity(4)
        transformation[:3, :3] = R
        transformation[:3, 3] = target_mean - np.dot(R, source_mean)
        
        # 소스 메쉬에 변환 적용
        source_model.mesh.transform(transformation)
        
        return transformation
    
    
        
