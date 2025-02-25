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
        """포인트 클라우드 전처리 - FPFH 계산 버전"""
        # 다운샘플링
        pcd_down = pcd.voxel_down_sample(voxel_size)
        
        # 법선 벡터 계산
        pcd_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
        
        # FPFH 특징점 계산
        pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            pcd_down,
            o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5, max_nn=100))
        
        return pcd_down, pcd_fpfh
    
    def preprocess_point_cloud_with_transform(self, pcd, voxel_size=0.08):
        """포인트 클라우드 전처리 (변환 행렬 포함)"""
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
        source_down, source_fpfh = self.preprocess_point_cloud(source_pcd, voxel_size)
        target_down, target_fpfh = self.preprocess_point_cloud(target_pcd, voxel_size)
        
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
        """PCA 정렬 수행"""
        try:
            print("PCA 정합 시작...")
            
            # 소스 포인트 클라우드 추출
            source_pcd = self.mesh_to_pointcloud(source_model.mesh)
            print("소스 포인트 클라우드 생성 완료")
            
            # 타겟 포인트 클라우드 추출
            target_pcd = self.mesh_to_pointcloud(target_model.mesh)
            print("타겟 포인트 클라우드 생성 완료")
            
            # PCA 계산 및 정렬
            print("PCA 계산 중...")
            # 소스 PCA
            source_pts = np.asarray(source_pcd.points)
            source_mean = np.mean(source_pts, axis=0)
            source_centered = source_pts - source_mean
            source_cov = np.cov(source_centered.T)
            source_eigvals, source_eigvecs = np.linalg.eigh(source_cov)
            # 고유값 순서 정렬 (내림차순)
            source_idx = np.argsort(source_eigvals)[::-1]
            source_eigvecs = source_eigvecs[:, source_idx]
            print("소스 PCA 계산 완료")
            
            # 타겟 PCA
            target_pts = np.asarray(target_pcd.points)
            target_mean = np.mean(target_pts, axis=0)
            target_centered = target_pts - target_mean
            target_cov = np.cov(target_centered.T)
            target_eigvals, target_eigvecs = np.linalg.eigh(target_cov)
            # 고유값 순서 정렬 (내림차순)
            target_idx = np.argsort(target_eigvals)[::-1]
            target_eigvecs = target_eigvecs[:, target_idx]
            print("타겟 PCA 계산 완료")
            
            # 회전 행렬 계산
            rotation = np.dot(source_eigvecs, target_eigvecs.T)
            
            # 변환 행렬 생성
            transformation = np.eye(4)
            transformation[:3, :3] = rotation
            # 먼저 소스의 중심을 원점으로 이동
            translation1 = np.eye(4)
            translation1[:3, 3] = -source_mean
            # 회전 후 타겟의 중심으로 이동
            translation2 = np.eye(4)
            translation2[:3, 3] = target_mean
            
            # 최종 변환 행렬: T2 * R * T1
            transformation = np.dot(translation2, np.dot(transformation, translation1))
            print("변환 행렬 계산 완료")
            
            # 소스 모델 변환
            source_model.transform(transformation)
            print("PCA 정합 완료")
            
            return transformation
            
        except Exception as e:
            print(f"PCA 정합 오류: {e}")
            import traceback
            traceback.print_exc()
            raise

    def compute_fpfh_features(self, source_model, target_model):
        """FPFH 특징 계산 및 시각화"""
        try:
            print("FPFH 특징점 추출 시작...")
            
            # 모델 확인
            if source_model is None or target_model is None:
                print("유효하지 않은 모델")
                raise ValueError("유효하지 않은 모델")
                
            # 메쉬를 포인트 클라우드로 변환
            print("소스 포인트 클라우드 추출 중...")
            source_pcd = self.mesh_to_pointcloud(source_model.mesh)
            print(f"소스 포인트 클라우드 생성 완료: {len(source_pcd.points)}점")
            
            print("타겟 포인트 클라우드 추출 중...")
            target_pcd = self.mesh_to_pointcloud(target_model.mesh)
            print(f"타겟 포인트 클라우드 생성 완료: {len(target_pcd.points)}점")
            
            # 다운샘플링 및 FPFH 계산 파라미터
            # 치아 모델에 적합한 값으로 조정
            self.voxel_size = 0.001  # 더 조밀한 다운샘플링 (치아 크기에 맞게)
            
            # 소스 모델 다운샘플링 - 변환 없는 버전 사용
            print("소스 포인트 클라우드 다운샘플링 중...")
            source_down, source_fpfh = self.preprocess_point_cloud(source_pcd, self.voxel_size)
            print(f"소스 다운샘플링 완료: {len(source_down.points)}점")
            
            # 타겟 모델 다운샘플링 - 변환 없는 버전 사용
            print("타겟 포인트 클라우드 다운샘플링 중...")
            target_down, target_fpfh = self.preprocess_point_cloud(target_pcd, self.voxel_size)
            print(f"타겟 다운샘플링 완료: {len(target_down.points)}점")
            
            # 특징점 시각화 준비
            source_sphere_size = self.voxel_size * 50
            target_sphere_size = self.voxel_size * 50
            
            # 소스 특징점 포인트 클라우드 생성
            print("소스 특징점 시각화 준비 중...")
            self.source_down = o3d.geometry.PointCloud()
            self.source_down.points = source_down.points
            self.source_down.paint_uniform_color([0.0, 0.5, 1.0])  # 밝은 파란색
            
            # 타겟 특징점 포인트 클라우드 생성
            print("타겟 특징점 시각화 준비 중...")
            self.target_down = o3d.geometry.PointCloud()
            self.target_down.points = target_down.points
            self.target_down.paint_uniform_color([1.0, 0.3, 0.3])  # 밝은 빨간색
            
            # FPFH 데이터 저장
            self.source_fpfh = source_fpfh
            self.target_fpfh = target_fpfh
            
            print("FPFH 특징점 추출 완료")
            
            # 결과 데이터 반환
            result = {
                'source_points': len(source_down.points),
                'target_points': len(target_down.points),
                'source_key_ratio': len(source_down.points) / len(source_pcd.points),
                'target_key_ratio': len(target_down.points) / len(target_pcd.points)
            }
            
            return result
            
        except Exception as e:
            print(f"FPFH 특징점 추출 오류: {e}")
            import traceback
            traceback.print_exc()
            raise
        
        
