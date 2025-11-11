# -*- coding: utf-8 -*-
import os
import sys
import glob
import pandas as pd
import geopandas as gpd
from datetime import datetime

# --- 1. 사용자 설정 부분 ---
QGIS_INSTALL_PATH = 'C:/Program Files/QGIS 3.40.11'  # 버전 업데이트 반영
VRA_FOLDER = 'data/sc/sc_vra_data'
BEFORE_FOLDER = 'data/sc/sc_before_data'
AFTER_FOLDER = 'data/sc/sc_after_data'
OUTPUT_IMAGE_FOLDER = 'data/sc/result_images'
OUTPUT_CSV_FOLDER = 'data/sc/result_csv'
OUTPUT_TEMP_FOLDER = 'data/sc/temp_layers'
OUTPUT_LOG_FOLDER = 'logs'
vra_area = '순창'

# 좌표계 설정 (모든 데이터를 이 좌표계로 통일)
TARGET_EPSG = 'EPSG:4326'  # WGS84 경위도 좌표계


# -------------------------

def setup_qgis_environment():
    """QGIS 환경을 설정하는 함수"""
    sys.path.append(os.path.join(QGIS_INSTALL_PATH, 'apps/qgis-ltr/python'))
    sys.path.append(os.path.join(QGIS_INSTALL_PATH, 'apps/qgis-ltr/python/plugins'))
    os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = os.path.join(QGIS_INSTALL_PATH, 'apps/Qt5/plugins')
    os.environ['QT_PLUGIN_PATH'] = os.path.join(QGIS_INSTALL_PATH, 'apps/qgis-ltr/qtplugins')
    print("QGIS 환경 설정 완료.")


def find_data_file(field_code, data_folder):
    """필지코드로 파일을 검색하되, 여러 개일 경우 GNDVI 파일을 우선 반환"""
    search_path = os.path.join(data_folder, f"{field_code}*.tif")
    files = glob.glob(search_path)
    if not files:
        print(f"   [오류] '{data_folder}'에서 '{field_code}'로 시작하는 파일을 찾을 수 없습니다.")
        return None
    if len(files) == 1:
        return files[0]
    for f in files:
        if 'GNDVI' in os.path.basename(f).upper():
            return f
    print(f"   [경고] '{data_folder}'에 '{field_code}' 파일이 여러 개지만 GNDVI 파일을 찾지 못했습니다. 첫 번째 파일을 사용합니다: {files[0]}")
    return files[0]


def reproject_raster_to_target(raster_path, output_path, target_epsg=TARGET_EPSG):
    """래스터 파일을 목표 좌표계로 재투영"""
    from qgis.core import QgsRasterLayer, QgsCoordinateReferenceSystem
    import processing

    raster = QgsRasterLayer(raster_path, "temp_raster")
    if not raster.isValid():
        print(f"   [오류] 래스터 파일을 불러올 수 없습니다: {raster_path}")
        return None

    source_crs = raster.crs().authid()
    print(f"   [좌표계 확인] 원본 래스터: {source_crs}")

    if source_crs == target_epsg:
        print(f"   [좌표계] 이미 {target_epsg}입니다. 변환 생략.")
        return raster_path

    print(f"   [좌표계 변환] {source_crs} → {target_epsg}")

    try:
        processing.run("gdal:warpreproject", {
            'INPUT': raster_path,
            'SOURCE_CRS': None,  # 원본 CRS 자동 감지
            'TARGET_CRS': QgsCoordinateReferenceSystem(target_epsg),
            'RESAMPLING': 0,  # Nearest Neighbour (DN 값 유지)
            'NODATA': None,
            'TARGET_RESOLUTION': None,
            'OPTIONS': '',
            'DATA_TYPE': 0,
            'TARGET_EXTENT': None,
            'TARGET_EXTENT_CRS': None,
            'MULTITHREADING': False,
            'EXTRA': '',
            'OUTPUT': output_path
        })
        print(f"   [성공] 좌표계 변환 완료: {output_path}")
        return output_path
    except Exception as e:
        print(f"   [오류] 좌표계 변환 실패: {e}")
        return None


def check_and_reproject_vector(vector_layer, target_epsg=TARGET_EPSG):
    """벡터 레이어의 좌표계 확인 및 경고"""
    from qgis.core import QgsCoordinateReferenceSystem

    source_crs = vector_layer.crs().authid()
    print(f"   [좌표계 확인] 벡터 레이어: {source_crs}")

    if source_crs != target_epsg:
        print(f"   [경고] 벡터 레이어가 {source_crs}입니다. 구역 통계 시 좌표계 불일치 가능성이 있습니다.")
        print(f"   [권장] 래스터를 {target_epsg}로 먼저 변환하면 벡터도 동일한 좌표계로 생성됩니다.")

    return vector_layer


def get_categorized_renderer(layer, field_name):
    """(1, 2단계용) 5개의 고유 DN 값을 찾아 범주형(Categorized) 렌더러 생성 (요구사항 반영)"""
    from qgis.core import QgsCategorizedSymbolRenderer, QgsSymbol, QgsRendererCategory
    from PyQt5.QtGui import QColor

    # DN 값 0을 제외하고 내림차순 정렬
    unique_values = layer.uniqueValues(layer.fields().indexFromName(field_name))
    dn_values = sorted([val for val in unique_values if val != 0], reverse=True)

    if len(dn_values) > 5:
        print(f"   [경고] DN 값이 5개 이상입니다. 상위 5개 값만 사용합니다: {dn_values[:5]}")
        dn_values = dn_values[:5]
    elif len(dn_values) < 5:
        print(f"   [경고] DN 값이 5개 미만입니다 (총 {len(dn_values)}개).")

    # 요구사항 색상: 1(빨강) ~ 5(파랑)
    colors = ['#d7191c', '#fdae61', '#ffffbf', '#abdda4', '#2b83ba']  # 높(빨) ~ 낮(파)
    categories = []

    for i, value in enumerate(dn_values):
        symbol = QgsSymbol.defaultSymbol(layer.geometryType())
        if i < len(colors):
            symbol.setColor(QColor(colors[i]))
        else:
            # 5개가 넘어가거나 모자랄 경우를 대비한 기본값
            symbol.setColor(QColor("#808080"))
        category = QgsRendererCategory(value, symbol, str(value))
        categories.append(category)

    renderer = QgsCategorizedSymbolRenderer(field_name, categories)
    return renderer


def set_labeling(layer, field_name, format_type='decimal'):
    """레이어에 라벨을 설정하는 함수 (정수 또는 소수점 3자리) (요구사항 반영)"""
    from qgis.core import QgsPalLayerSettings, QgsVectorLayerSimpleLabeling, QgsTextFormat

    layer_settings = QgsPalLayerSettings()
    text_format = QgsTextFormat()
    layer_settings.setFormat(text_format)

    if format_type == 'integer':
        # 정수 표시 (DN 값)
        expression = f"format_number( \"{field_name}\", 0)"
    else:
        # 소수점 3자리 표시 (before_mean, after_mean)
        expression = f"format_number( \"{field_name}\", 3)"

    layer_settings.fieldName = expression
    layer_settings.isExpression = True

    labeling = QgsVectorLayerSimpleLabeling(layer_settings)
    layer.setLabelsEnabled(True)
    layer.setLabeling(labeling)


def export_styled_layer_to_image(layer, output_path, scale=None):
    """스타일이 적용된 벡터 레이어를 직접 렌더링하여 PNG로 저장합니다. (요구사항 반영)"""
    from qgis.core import QgsMapSettings, QgsMapRendererParallelJob, QgsRectangle
    from PyQt5.QtCore import QSize, QEventLoop
    from PyQt5.QtGui import QColor

    map_settings = QgsMapSettings()
    map_settings.setLayers([layer])
    map_settings.setBackgroundColor(QColor(255, 255, 255, 0))  # 배경 투명
    extent = layer.extent()

    if scale:
        # [3단계] 요구사항: 축척 1:150 적용
        # QGIS API: extent와 outputSize를 조정하여 축척 구현

        # A4 가로 (297x210mm) 기준 크기
        width_mm = 297
        height_mm = 210
        dpi = 150

        # 픽셀 크기 계산
        width_px = int(width_mm / 25.4 * dpi)  # 약 1754px
        height_px = int(height_mm / 25.4 * dpi)  # 약 1240px

        # 축척 1:150을 반영한 실제 지도 범위 계산
        # 축척 = 지도상 거리 / 실제 거리
        # 1:150 = 1mm on map = 150mm in reality = 0.15m

        # 이미지 너비(픽셀) → 실제 지도 너비(미터)
        map_width_m = (width_mm / 1000.0) * scale  # 297mm * 150 = 44.55m
        map_height_m = (height_mm / 1000.0) * scale  # 210mm * 150 = 31.5m

        # extent의 중심점
        center_x = extent.center().x()
        center_y = extent.center().y()

        # 새로운 extent 계산 (축척을 반영)
        new_extent = QgsRectangle(
            center_x - map_width_m / 2,
            center_y - map_height_m / 2,
            center_x + map_width_m / 2,
            center_y + map_height_m / 2
        )

        map_settings.setExtent(new_extent)
        map_settings.setOutputSize(QSize(width_px, height_px))

        print(f"   [축척 정보] 1:{scale}, 지도 크기: {map_width_m:.2f}m × {map_height_m:.2f}m, 이미지: {width_px}×{height_px}px")
    else:
        # [1, 2단계] 축척 미지정: 레이어 영역에 맞춤
        map_settings.setExtent(extent)
        width_px = 2000  # 기본 너비
        height_px = int(width_px * extent.height() / extent.width())
        map_settings.setOutputSize(QSize(width_px, height_px))

    job = QgsMapRendererParallelJob(map_settings)
    loop = QEventLoop()
    job.finished.connect(loop.quit)
    job.start()
    loop.exec_()

    image = job.renderedImage()
    image.save(output_path, "png")


def main():
    """메인 실행 함수"""
    start_time = datetime.now()
    print("=" * 80)
    print("QGIS 변량시비 자동화 스크립트 실행 시작")
    print(f"시작 시각: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    setup_qgis_environment()

    # QgsEqualIntervalClassification Import 제거 (모듈 오류 방지)
    from qgis.core import (QgsApplication, QgsVectorLayer, QgsRasterLayer,
                           QgsProject, QgsSymbol, QgsGraduatedSymbolRenderer,
                           QgsStyle)
    from PyQt5.QtCore import QEventLoop, QSize
    from PyQt5.QtGui import QColor

    from qgis.analysis import QgsNativeAlgorithms  # QgsEqualIntervalClassification Import 제거
    import processing

    qgs = QgsApplication([], False)
    qgs.initQgis()

    sys.path.append(os.path.join(QGIS_INSTALL_PATH, 'apps/qgis-ltr/python/plugins'))
    from processing.core.Processing import Processing
    Processing.initialize()
    QgsApplication.processingRegistry().addProvider(QgsNativeAlgorithms())

    project = QgsProject.instance()

    # 출력 폴더 생성
    for folder in [OUTPUT_IMAGE_FOLDER, OUTPUT_CSV_FOLDER, OUTPUT_TEMP_FOLDER, OUTPUT_LOG_FOLDER]:
        if not os.path.exists(folder):
            os.makedirs(folder)
            print(f"[폴더 생성] {folder}")

    cv_data_list = []
    growth_data_list = []

    # 처리 현황 추적
    processing_log = {
        'success': [],
        'failed': [],
        'total': 0
    }

    # [1-1] VRA 파일 불러오기
    vra_rx_files = glob.glob(os.path.join(VRA_FOLDER, '*_Rx.tif'))
    if not vra_rx_files:
        print(f"[오류] '{VRA_FOLDER}'에 '*_Rx.tif' 파일이 없습니다.")
        qgs.exitQgis()
        return

    processing_log['total'] = len(vra_rx_files)
    print(f"\n총 {len(vra_rx_files)}개의 필지를 처리합니다.")
    print("=" * 80)

    for idx, vra_rx_file in enumerate(vra_rx_files, 1):
        filename = os.path.basename(vra_rx_file)
        field_code = filename.split('_')[0]
        print(f"\n{'=' * 80}")
        print(f"[{idx}/{len(vra_rx_files)}] 필지: {field_code}")
        print(f"{'=' * 80}")

        try:
            # === 좌표계 변환: VRA 래스터 ===
            print("   [0단계] VRA 래스터 좌표계 변환...")
            reprojected_vra_path = os.path.join(OUTPUT_TEMP_FOLDER, f"{field_code}_vra_reprojected.tif")
            vra_rx_file_to_use = reproject_raster_to_target(vra_rx_file, reprojected_vra_path)

            if vra_rx_file_to_use is None:
                raise Exception("VRA 래스터 좌표계 변환 실패")

            # === 1단계: VRA 처방맵 가공 ===
            print("   [1단계] VRA 래스터 벡터화 및 스타일링...")
            vector_path = os.path.join(OUTPUT_TEMP_FOLDER, f"{field_code}_vra_vector.gpkg")

            # [1-2] 벡터 변환 (DN 필드)
            processing.run("gdal:polygonize", {
                'INPUT': vra_rx_file_to_use, 'BAND': 1, 'FIELD': 'DN',
                'EIGHT_CONNECTEDNESS': False, 'OUTPUT': vector_path
            })

            vra_vector_layer = QgsVectorLayer(vector_path, f"{field_code}_vra_vector", "ogr")
            if not vra_vector_layer.isValid():
                raise Exception(f"벡터 변환 실패: {vector_path}")

            # 벡터 좌표계 확인
            check_and_reproject_vector(vra_vector_layer)

            # [1-3] DN=0 필드 삭제
            vra_vector_layer.startEditing()
            deleted_count = vra_vector_layer.deleteFeatures(
                [f.id() for f in vra_vector_layer.getFeatures() if f['DN'] == 0])
            vra_vector_layer.commitChanges()
            print(f"   [정리] DN=0 피처 {deleted_count}개 삭제")

            # [1-3-1] 면적(ha) 및 비료 살포량(spray) 계산
            print("   [전처리] 면적 및 비료 살포량 계산...")
            vra_vector_layer.startEditing()

            # area(ha) 필드 추가 (중복 체크)
            from qgis.core import QgsField, QgsDistanceArea, QgsCoordinateReferenceSystem, QgsCoordinateTransformContext
            from PyQt5.QtCore import QVariant

            existing_fields = [field.name() for field in vra_vector_layer.fields()]
            new_fields = []
            if 'area_ha' not in existing_fields:
                new_fields.append(QgsField('area_ha', QVariant.Double))
            if 'spray' not in existing_fields:
                new_fields.append(QgsField('spray', QVariant.Double))

            if new_fields:
                vra_vector_layer.dataProvider().addAttributes(new_fields)
                vra_vector_layer.updateFields()

            # QgsDistanceArea 설정 (타원체 기반 면적 계산)
            distance_area = QgsDistanceArea()
            distance_area.setSourceCrs(vra_vector_layer.crs(), QgsCoordinateTransformContext())
            distance_area.setEllipsoid('WGS84')  # WGS84 타원체 사용

            # 각 피처의 면적 및 살포량 계산
            area_ha_idx = vra_vector_layer.fields().indexFromName('area_ha')
            spray_idx = vra_vector_layer.fields().indexFromName('spray')
            dn_idx = vra_vector_layer.fields().indexFromName('DN')

            print(f"   [디버깅] 필드 인덱스 - area_ha: {area_ha_idx}, spray: {spray_idx}, DN: {dn_idx}")
            print(f"   [디버깅] 레이어 CRS: {vra_vector_layer.crs().authid()}")
            print(f"   [디버깅] 타원체 기반 면적 계산 사용 (QgsDistanceArea)")

            feature_count = 0
            total_spray = 0
            for feature in vra_vector_layer.getFeatures():
                # 타원체 기반 면적 계산 (m² 단위로 정확하게 계산)
                area_m2 = distance_area.measureArea(feature.geometry())
                area_ha = area_m2 / 10000.0

                # 비료 살포량 = DN * area(ha)
                dn_value = feature[dn_idx]
                spray_value = dn_value * area_ha

                # 디버깅 출력 (처음 3개 피처만)
                if feature_count < 3:
                    print(
                        f"   [디버깅] 피처 #{feature.id()}: DN={dn_value}, area_m2={area_m2:.2f}, area_ha={area_ha:.6f}, spray={spray_value:.6f}")

                # 필드 값 업데이트
                vra_vector_layer.changeAttributeValue(feature.id(), area_ha_idx, area_ha)
                vra_vector_layer.changeAttributeValue(feature.id(), spray_idx, spray_value)

                feature_count += 1
                total_spray += spray_value

            vra_vector_layer.commitChanges()
            print(f"   [성공] 총 {feature_count}개 피처 처리 완료")
            print(f"   [통계] 총 비료 살포량: {total_spray:.3f}")

            # 계산 결과 검증 (필드 값이 실제로 저장되었는지 확인)
            print(f"   [검증] 저장된 값 확인 (처음 3개 피처):")
            verify_count = 0
            for feature in vra_vector_layer.getFeatures():
                if verify_count < 3:
                    dn_val = feature['DN']
                    area_val = feature['area_ha']
                    spray_val = feature['spray']
                    print(f"   [검증] 피처 #{feature.id()}: DN={dn_val}, area_ha={area_val}, spray={spray_val}")
                    verify_count += 1
                else:
                    break

            # [1-4] 심볼 변경 (Categorized - DN 기준 색상)
            vra_vector_layer.setRenderer(get_categorized_renderer(vra_vector_layer, "DN"))
            # [1-5] 라벨 표시 (spray, 소수점 3자리)
            set_labeling(vra_vector_layer, "spray", format_type='decimal')

            # [1-6] 이미지로 저장
            img_path_1 = os.path.join(OUTPUT_IMAGE_FOLDER, f"{field_code}_VRA.png")
            export_styled_layer_to_image(vra_vector_layer, img_path_1)
            print(f"   [성공] 1단계 이미지 저장: {img_path_1}")

            # === 2단계: 'Before' TIF 구역 통계 ===
            print("   [2단계] 'Before' TIF 구역 통계 및 스타일링...")
            # [2-1] Before TIF 파일 불러오기
            before_raster_path_original = find_data_file(field_code, BEFORE_FOLDER)
            if not before_raster_path_original:
                raise Exception("Before TIF 파일을 찾을 수 없습니다")

            # Before 래스터 좌표계 변환
            reprojected_before_path = os.path.join(OUTPUT_TEMP_FOLDER, f"{field_code}_before_reprojected.tif")
            before_raster_path = reproject_raster_to_target(before_raster_path_original, reprojected_before_path)

            if before_raster_path is None:
                raise Exception("Before 래스터 좌표계 변환 실패")

            before_stats_path = os.path.join(OUTPUT_TEMP_FOLDER, f"{field_code}_vra_before.gpkg")

            # [2-2] 구역 통계 (접두어 'before_')
            # [★개선사항] STATISTICS: Mean(2), Median(3), Min(5), Max(6), StdDev(4)
            processing.run("native:zonalstatisticsfb", {
                'INPUT_RASTER': before_raster_path, 'INPUT': vector_path, 'BAND': 1,
                'COLUMN_PREFIX': 'before_', 'STATISTICS': [2, 3, 5, 6, 4],
                'OUTPUT': before_stats_path
            })

            # [2-3] 레이어 이름 설정
            vra_before_layer = QgsVectorLayer(before_stats_path, f"{field_code}_vra_before", "ogr")
            if not vra_before_layer.isValid():
                raise Exception(f"'Before' 구역 통계 실패: {before_stats_path}")

            # [2-4] 심볼 설정 (1단계와 동일)
            vra_before_layer.setRenderer(get_categorized_renderer(vra_before_layer, "DN"))
            # [2-5] 라벨 표시 (before_mean, 소수점 3자리)
            set_labeling(vra_before_layer, "before_mean", format_type='decimal')

            # [2-6] 이미지로 저장
            img_path_2 = os.path.join(OUTPUT_IMAGE_FOLDER, f"{field_code}_VRA_before.png")
            export_styled_layer_to_image(vra_before_layer, img_path_2)
            print(f"   [성공] 2단계 이미지 저장: {img_path_2}")

            # === 3단계: 'After' TIF 구역 통계 ===
            print("   [3단계] 'After' TIF 구역 통계 실행...")
            # [3-1] After TIF 파일 불러오기
            after_raster_path_original = find_data_file(field_code, AFTER_FOLDER)
            if not after_raster_path_original:
                raise Exception("After TIF 파일을 찾을 수 없습니다")

            # After 래스터 좌표계 변환
            reprojected_after_path = os.path.join(OUTPUT_TEMP_FOLDER, f"{field_code}_after_reprojected.tif")
            after_raster_path = reproject_raster_to_target(after_raster_path_original, reprojected_after_path)

            if after_raster_path is None:
                raise Exception("After 래스터 좌표계 변환 실패")

            # [3-3] 산출물 이름 설정
            after_stats_path = os.path.join(OUTPUT_TEMP_FOLDER, f"{field_code}_vra_after.gpkg")

            # [3-2] 구역 통계 (입력: vra_before, 접두어: 'after_')
            # [★개선사항] STATISTICS: Mean(2), Median(3), Min(5), Max(6), StdDev(4)
            processing.run("native:zonalstatisticsfb", {
                'INPUT_RASTER': after_raster_path, 'INPUT': before_stats_path, 'BAND': 1,
                'COLUMN_PREFIX': 'after_', 'STATISTICS': [2, 3, 5, 6, 4],
                'OUTPUT': after_stats_path
            })

            vra_after_layer = QgsVectorLayer(after_stats_path, f"{field_code}_vra_after", "ogr")
            if not vra_after_layer.isValid():
                raise Exception(f"'After' 구역 통계 실패: {after_stats_path}")

            # === 4단계: CSV 데이터 집계 (요구사항 순서: 이미지 생성 전) ===
            print("   [4단계] CSV 데이터 집계...")
            gdf = gpd.read_file(after_stats_path)

            # [4-1] 기본 통계 (표준편차)
            before_mean_std = gdf['before_mean'].std()
            after_mean_std = gdf['after_mean'].std()
            cv_data_list.append({
                '필지코드': field_code,
                'before_mean_cv': before_mean_std,  # 요구사항 컬럼명 'before_mean_cv'
                'after_mean_cv': after_mean_std  # 요구사항 컬럼명 'after_mean_cv'
            })

            # [4-2] 영역별 데이터 (그룹별)
            # DN 그룹과 색상 이름 매핑 (1단계와 동일한 로직)
            unique_dns_sorted = sorted([val for val in gdf['DN'].unique() if val != 0], reverse=True)
            dn_color_names = ['빨강색', '주황색', '베이지색', '초록색', '파란색']
            dn_to_color_map = {}
            for i, dn_val in enumerate(unique_dns_sorted):
                if i < len(dn_color_names):
                    dn_to_color_map[dn_val] = dn_color_names[i]
                else:
                    dn_to_color_map[dn_val] = '기타'  # 5개 초과시

            # DN별 그룹화 및 통계 계산
            print(f"   [디버깅] GeoDataFrame 행 수: {len(gdf)}")
            print(f"   [디버깅] 고유 DN 값: {sorted(gdf['DN'].unique())}")

            grouped = gdf.groupby('DN')
            dn_count = 0
            for dn_value, group in grouped:
                if dn_value == 0:
                    print(f"   [경고] DN=0 피처 발견 ({len(group)}개) - 건너뜀")
                    continue  # DN=0은 이미 삭제했지만 안전장치

                # DN별 피처 수 확인
                feature_count_in_group = len(group)

                before_mean_result = group['before_mean'].mean()
                after_mean_result = group['after_mean'].mean()

                # DN별 면적(ha) 합계 계산
                area_ha_sum = group['area_ha'].sum()

                # DN별 비료 살포량(spray) 합계 계산
                spray_sum = group['spray'].sum()

                vi_rate = 0
                if before_mean_result != 0 and before_mean_result is not None:
                    # 증감율 계산
                    vi_rate = (after_mean_result - before_mean_result) / before_mean_result * 100

                # color_group 컬럼 생성
                color_group_name = dn_to_color_map.get(dn_value, '알 수 없음')

                # 디버깅 출력
                print(
                    f"   [DN={dn_value}] 피처 수: {feature_count_in_group}, 면적: {area_ha_sum:.6f}ha, 살포량: {spray_sum:.3f}")

                growth_data_list.append({
                    '필지코드': field_code,
                    'DN': int(dn_value),
                    'color_group': color_group_name,
                    'area_ha': area_ha_sum,
                    'spray_sum': spray_sum,
                    'before_mean_result': before_mean_result,
                    'after_mean_result': after_mean_result,
                    'vi_rate': vi_rate
                })
                dn_count += 1

            print(f"   [성공] {field_code} 필지 통계 집계 완료 (총 {dn_count}개 DN 그룹)")

            # === 5단계 (구 3단계): 'After' 이미지 생성 ===
            print("   [5단계] 'After' 레이어 스타일링 및 이미지 저장...")

            from qgis.core import QgsGraduatedSymbolRenderer, QgsRendererRange, QgsClassificationEqualInterval

            # 기본 심볼 및 색상 램프 설정
            symbol = QgsSymbol.defaultSymbol(vra_after_layer.geometryType())
            color_ramp = QgsStyle.defaultStyle().colorRamp('Spectral')

            # after_mean 필드의 값 범위 계산
            field_index = vra_after_layer.fields().indexFromName('after_mean')
            provider = vra_after_layer.dataProvider()
            min_value = provider.minimumValue(field_index)
            max_value = provider.maximumValue(field_index)

            # Equal Interval 분류로 5개 등급 생성
            num_classes = 5
            interval = (max_value - min_value) / num_classes

            ranges = []
            for i in range(num_classes):
                lower = min_value + (i * interval)
                upper = min_value + ((i + 1) * interval)

                # 심볼 생성 및 색상 설정
                range_symbol = symbol.clone()
                # Spectral 색상 램프에서 색상 가져오기 (0.0 ~ 1.0)
                color = color_ramp.color(i / (num_classes - 1))
                range_symbol.setColor(color)

                # 범주 레이블 생성
                label = f"{lower:.3f} - {upper:.3f}"

                # QgsRendererRange 생성
                renderer_range = QgsRendererRange(lower, upper, range_symbol, label)
                ranges.append(renderer_range)

            # Graduated Symbol Renderer 생성
            renderer = QgsGraduatedSymbolRenderer('after_mean', ranges)
            renderer.setMode(QgsGraduatedSymbolRenderer.EqualInterval)

            vra_after_layer.setRenderer(renderer)

            # [3-5] 라벨 설정 (after_mean, 소수점 3자리)
            set_labeling(vra_after_layer, "after_mean", format_type='decimal')

            # [3-6] 이미지 저장 (전체 레이어 extent 사용 - 축척 파라미터 제거)
            # 주의: scale=150을 적용하면 44.55m × 31.5m extent가 생성되어 피처가 잘릴 수 있습니다.
            # 전체 레이어를 표시하기 위해 scale 파라미터를 제거합니다.
            img_path_3 = os.path.join(OUTPUT_IMAGE_FOLDER, f"{field_code}_VRA_after.png")
            export_styled_layer_to_image(vra_after_layer, img_path_3)
            print(f"   [성공] 5단계(After) 이미지 저장: {img_path_3}")

            # 레이어 객체 삭제 (메모리 관리)
            del vra_vector_layer
            del vra_before_layer
            del vra_after_layer

            # 성공 기록
            processing_log['success'].append(field_code)
            print(f"   [완료] {field_code} 필지 처리 성공 ✓")

        except Exception as e:
            # 실패 기록
            processing_log['failed'].append({'field_code': field_code, 'error': str(e)})
            print(f"   [실패] {field_code} 필지 처리 실패: {e}")
            continue

    print("\n" + "=" * 80)
    print("모든 필지 처리 완료")
    print("=" * 80)

    # === 6단계: 최종 CSV 파일 저장 (루프 밖) ===
    print("\n[6단계] 최종 CSV 파일 저장...")

    # [4-1] CSV 저장
    csv_path_1 = os.path.join(OUTPUT_CSV_FOLDER, f'{vra_area}_변량시비 전후 표준편차.csv')
    df1 = pd.DataFrame(cv_data_list)
    df1.to_csv(csv_path_1, index=False, encoding='utf-8-sig')
    print(f"[최종 성공] CV 통계 CSV 파일 저장 완료: {csv_path_1}")
    print(f"   - 총 {len(df1)}개 필지 데이터")

    # [4-2] CSV 저장
    csv_path_2 = os.path.join(OUTPUT_CSV_FOLDER, f'{vra_area}_변량시비 전후 생육 변화량 확인.csv')
    df2 = pd.DataFrame(growth_data_list)
    df2.to_csv(csv_path_2, index=False, encoding='utf-8-sig')
    print(f"[최종 성공] 생육 변화량 CSV 파일 저장 완료: {csv_path_2}")
    print(f"   - 총 {len(df2)}개 영역 데이터")

    # === 7단계: 처리 결과 보고서 생성 ===
    print("\n" + "=" * 80)
    print("처리 결과 요약")
    print("=" * 80)
    print(f"총 필지 수: {processing_log['total']}")
    print(f"성공: {len(processing_log['success'])}개")
    print(f"실패: {len(processing_log['failed'])}개")

    if processing_log['success']:
        print(f"\n✓ 성공한 필지: {', '.join(processing_log['success'])}")

    if processing_log['failed']:
        print(f"\n✗ 실패한 필지:")
        for failed in processing_log['failed']:
            print(f"   - {failed['field_code']}: {failed['error']}")

    # 로그 파일 저장
    end_time = datetime.now()
    duration = end_time - start_time

    log_filename = f"processing_log_{start_time.strftime('%Y%m%d_%H%M%S')}.txt"
    log_path = os.path.join(OUTPUT_LOG_FOLDER, log_filename)

    with open(log_path, 'w', encoding='utf-8') as log_file:
        log_file.write("=" * 80 + "\n")
        log_file.write("QGIS 변량시비 자동화 처리 로그\n")
        log_file.write("=" * 80 + "\n\n")
        log_file.write(f"시작 시각: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_file.write(f"종료 시각: {end_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_file.write(f"총 소요 시간: {duration}\n\n")
        log_file.write(f"총 필지 수: {processing_log['total']}\n")
        log_file.write(f"성공: {len(processing_log['success'])}개\n")
        log_file.write(f"실패: {len(processing_log['failed'])}개\n\n")

        if processing_log['success']:
            log_file.write("=" * 80 + "\n")
            log_file.write("성공한 필지\n")
            log_file.write("=" * 80 + "\n")
            for field_code in processing_log['success']:
                log_file.write(f"✓ {field_code}\n")

        if processing_log['failed']:
            log_file.write("\n" + "=" * 80 + "\n")
            log_file.write("실패한 필지\n")
            log_file.write("=" * 80 + "\n")
            for failed in processing_log['failed']:
                log_file.write(f"✗ {failed['field_code']}\n")
                log_file.write(f"   오류: {failed['error']}\n\n")

    print(f"\n[로그 저장] 처리 로그 파일 저장 완료: {log_path}")

    print("\n" + "=" * 80)
    print(f"모든 작업 완료 (총 소요 시간: {duration})")
    print("=" * 80)

    qgs.exitQgis()


if __name__ == '__main__':
    main()
