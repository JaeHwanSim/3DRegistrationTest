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

    def preprocess_point_cloud(self, pcd, voxel_size=0.03):
        """포인트 클라우드 전처리 개선"""
        # 다운샘플링
        pcd_down = pcd.voxel_down_sample(voxel_size)
        
        # 노이즈 제거
        pcd_down, _ = pcd_down.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        
        # 법선 벡터 계산 - 더 조밀한 탐색
        pcd_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=50))
        
        # 법선 벡터 방향 일관성 확보
        pcd_down.orient_normals_consistent_tangent_plane(k=30)
        
        # FPFH 특징점 계산 - 파라미터 조정
        pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            pcd_down,
            o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5, max_nn=150))
        
        return pcd_down, pcd_fpfh

    def estimate_rough_alignment(self, source_pcd, target_pcd):
        """대략적인 초기 정렬 추정"""
        # 중심점 계산
        source_center = np.mean(np.asarray(source_pcd.points), axis=0)
        target_center = np.mean(np.asarray(target_pcd.points), axis=0)
        
        # 주성분 분석 (PCA)
        source_pcd_centered = o3d.geometry.PointCloud()
        source_pcd_centered.points = o3d.utility.Vector3dVector(
            np.asarray(source_pcd.points) - source_center)
        target_pcd_centered = o3d.geometry.PointCloud()
        target_pcd_centered.points = o3d.utility.Vector3dVector(
            np.asarray(target_pcd.points) - target_center)
        
        # 공분산 행렬 계산
        source_covariance = np.cov(np.asarray(source_pcd_centered.points).T)
        target_covariance = np.cov(np.asarray(target_pcd_centered.points).T)
        
        # 주방향 계산
        source_eigenvalues, source_eigenvectors = np.linalg.eigh(source_covariance)
        target_eigenvalues, target_eigenvectors = np.linalg.eigh(target_covariance)
        
        # 회전 행렬 계산
        R = np.dot(target_eigenvectors, source_eigenvectors.T)
        
        # 변환 행렬 생성
        transformation = np.identity(4)
        transformation[:3, :3] = R
        transformation[:3, 3] = target_center - np.dot(R, source_center)
        
        return transformation

    def execute_global_registration(self, source_down, target_down, 
                                  source_fpfh, target_fpfh, voxel_size):
        """개선된 전역 정합"""
        distance_threshold = voxel_size * 1.5
        
        # 더 엄격한 RANSAC 파라미터 설정
        result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            source_down, target_down, source_fpfh, target_fpfh,
            True,
            distance_threshold,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
            4,  # 최소 대응점 개수 증가
            [
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnNormal(0.52)  # 법선 벡터 체크 추가
            ],
            o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 1000))
        return result

    def refine_registration(self, source, target, transformation, voxel_size):
        """다단계 ICP 정밀 정합"""
        current_transformation = transformation
        
        # 여러 단계의 ICP 실행 (거친 정합에서 정밀 정합으로)
        for threshold_multiplier in [2.0, 1.4, 1.0, 0.6]:
            distance_threshold = voxel_size * threshold_multiplier
            
            result = o3d.pipelines.registration.registration_icp(
                source, target, distance_threshold, current_transformation,
                o3d.pipelines.registration.TransformationEstimationPointToPlane(),  # 평면 기반 정합으로 변경
                o3d.pipelines.registration.ICPConvergenceCriteria(
                    relative_fitness=1e-6,
                    relative_rmse=1e-6,
                    max_iteration=100))
            
            current_transformation = result.transformation
            
        return result

    def register(self, source_model, target_model):
        """개선된 정합 프로세스"""
        # 포인트 클라우드 변환
        source_pcd = self.mesh_to_pointcloud(source_model.mesh)
        target_pcd = self.mesh_to_pointcloud(target_model.mesh)
        
        # 전처리
        voxel_size = 0.03  # 더 작은 복셀 사이즈
        source_down, source_fpfh = self.preprocess_point_cloud(source_pcd, voxel_size)
        target_down, target_fpfh = self.preprocess_point_cloud(target_pcd, voxel_size)
        
        # 초기 정렬 추정
        initial_transformation = self.estimate_rough_alignment(source_down, target_down)
        source_down.transform(initial_transformation)
        source_pcd.transform(initial_transformation)
        
        # 전역 정합
        result_global = self.execute_global_registration(
            source_down, target_down, source_fpfh, target_fpfh, voxel_size)
        
        # 정밀 정합
        result_icp = self.refine_registration(
            source_pcd, target_pcd, result_global.transformation, voxel_size)
        
        # 최종 변환 행렬 계산 (초기 정렬 + 전역 정합 + 정밀 정합)
        self.transformation = np.dot(result_icp.transformation, 
                                   np.dot(result_global.transformation, 
                                         initial_transformation))
        
        # 소스 메쉬에 변환 적용
        source_model.mesh.transform(self.transformation)
        
        return {
            'fitness': result_icp.fitness,
            'inlier_rmse': result_icp.inlier_rmse,
            'transformation': self.transformation,
            'global_fitness': result_global.fitness,
            'global_rmse': result_global.inlier_rmse
        }
    
    
        
