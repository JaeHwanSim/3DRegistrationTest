import open3d as o3d
import numpy as np
from scipy.spatial import KDTree

class Registration:
    def __init__(self):
        self.source = None
        self.target = None
        self.transformation = np.identity(4)
        
    def mesh_to_pointcloud(self, mesh, sample_density=50000):
        """TriangleMesh를 PointCloud로 변환
        
        Args:
            mesh: 변환할 메쉬
            sample_density: 샘플링할 포인트 수
        """
        # 메쉬 표면에서 균일하게 포인트 샘플링
        pcd = mesh.sample_points_uniformly(number_of_points=sample_density)
        
        # 법선 벡터 계산
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30))
        pcd.normalize_normals()
        
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

    def execute_fpfh_extraction(self, source_model, target_model, voxel_size=0.001):
        """FPFH 특징점 추출"""
        # 포인트 클라우드 변환
        source_pcd = self.mesh_to_pointcloud(source_model.mesh)
        target_pcd = self.mesh_to_pointcloud(target_model.mesh)
        
        print(f"초기 포인트 수 - 소스: {len(source_pcd.points)}, 타겟: {len(target_pcd.points)}")
        
        # 다운샘플링
        source_down = source_pcd.voxel_down_sample(voxel_size)
        target_down = target_pcd.voxel_down_sample(voxel_size)
        
        print(f"다운샘플링 후 포인트 수 - 소스: {len(source_down.points)}, 타겟: {len(target_down.points)}")
        
        # 법선 벡터 계산
        source_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5, max_nn=100))
        target_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5, max_nn=100))
        
        # FPFH 특징점 계산
        source_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            source_down,
            o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 10, max_nn=200))
        target_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            target_down,
            o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 10, max_nn=200))
        
        # 특징점의 중요도 계산
        source_importance = np.sum(np.abs(source_fpfh.data), axis=0)
        target_importance = np.sum(np.abs(target_fpfh.data), axis=0)
        
        # 상위 95% 특징점 선택
        source_threshold = np.percentile(source_importance, 5)
        target_threshold = np.percentile(target_importance, 5)
        
        source_key_indices = np.where(source_importance > source_threshold)[0]
        target_key_indices = np.where(target_importance > target_threshold)[0]
        
        # 특징점 시각화를 위한 포인트 클라우드 생성
        source_down_key = source_down.select_by_index(source_key_indices)
        target_down_key = target_down.select_by_index(target_key_indices)
        
        # 특징점을 구로 표현
        source_spheres = o3d.geometry.TriangleMesh()
        target_spheres = o3d.geometry.TriangleMesh()
        
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=voxel_size*10)
        sphere.compute_vertex_normals()
        
        # 시각화를 위해 최대 1000개의 점만 선택
        max_points = 1000
        source_points = np.asarray(source_down_key.points)
        target_points = np.asarray(target_down_key.points)
        
        if len(source_points) > max_points:
            source_indices = np.random.choice(len(source_points), max_points, replace=False)
            source_points = source_points[source_indices]
        
        if len(target_points) > max_points:
            target_indices = np.random.choice(len(target_points), max_points, replace=False)
            target_points = target_points[target_indices]
        
        for point in source_points:
            sphere_copy = o3d.geometry.TriangleMesh(sphere)
            sphere_copy.translate(point)
            sphere_copy.paint_uniform_color([0, 0, 1])  # 파란색
            source_spheres += sphere_copy
            
        for point in target_points:
            sphere_copy = o3d.geometry.TriangleMesh(sphere)
            sphere_copy.translate(point)
            sphere_copy.paint_uniform_color([1, 0, 0])  # 빨간색
            target_spheres += sphere_copy
        
        print(f"전체 특징점 수 - 소스: {len(source_key_indices)}, 타겟: {len(target_key_indices)}")
        print(f"시각화된 특징점 수 - 소스: {len(source_points)}, 타겟: {len(target_points)}")
        
        # 결과 저장
        self.source_down = source_spheres
        self.target_down = target_spheres
        self.source_fpfh = source_fpfh
        self.target_fpfh = target_fpfh
        
        return {
            'source_points': len(source_key_indices),
            'target_points': len(target_key_indices),
            'source_fpfh': source_fpfh.data.shape,
            'target_fpfh': target_fpfh.data.shape,
            'source_key_ratio': len(source_key_indices) / len(source_importance),
            'target_key_ratio': len(target_key_indices) / len(target_importance)
        }
        
        
