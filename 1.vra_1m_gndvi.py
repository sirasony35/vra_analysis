import os
import glob
import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import reproject, Resampling
import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib.patches as mpatches

# =========================================================
# [설정 영역] 사용자 경로 및 파라미터
# =========================================================
DIR_BEFORE = r"D:\회사관련\VRA_analysis\before_data"  # 기준 (Before)
DIR_AFTER = r"D:\회사관련\VRA_analysis\after_data"  # 비교 (After)
DIR_OUTPUT = r"D:\회사관련\VRA_analysis\Result_Images"  # 결과 저장 폴더

# 목표 해상도 (1m)
TARGET_GSD = 1.0

# 색상 체계 (Zone 1 ~ Zone 5)
COLOR_LIST = ['#c51f1e', '#f5a361', '#faf7be', '#a1d193', '#447cb9']
ZONE_NAMES = ["Zone 1", "Zone 2", "Zone 3", "Zone 4", "Zone 5"]


def get_field_codes(directory):
    if not os.path.exists(directory):
        print(f"[오류] 폴더 없음: {directory}")
        return []
    files = glob.glob(os.path.join(directory, "*.tif"))
    codes = set()
    for f in files:
        fname = os.path.basename(f)
        code = fname.split('_')[0] if '_' in fname else os.path.splitext(fname)[0]
        codes.add(code)
    return sorted(list(codes))


def find_file(directory, code):
    pattern = os.path.join(directory, f"*{code}*.tif")
    files = glob.glob(pattern)
    if not files: return None
    gndvi = [f for f in files if "GNDVI" in os.path.basename(f).upper()]
    return gndvi[0] if gndvi else files[0]


def read_and_resample(path, target_gsd=1.0):
    with rasterio.open(path) as src:
        scale_x = src.res[0] / target_gsd
        scale_y = src.res[1] / target_gsd
        new_height = int(src.height * scale_y)
        new_width = int(src.width * scale_x)

        new_transform = src.transform * src.transform.scale(
            (src.width / new_width), (src.height / new_height)
        )

        data = src.read(
            1,
            out_shape=(new_height, new_width),
            resampling=Resampling.bilinear
        )

        profile = src.profile.copy()
        profile.update({'height': new_height, 'width': new_width, 'transform': new_transform})

        mask = (data > 0) & (~np.isnan(data))
        if src.nodata is not None:
            mask = mask & (data != src.nodata)

        return data, mask, profile


def align_image(src_path, ref_profile):
    with rasterio.open(src_path) as src:
        dst_height = ref_profile['height']
        dst_width = ref_profile['width']
        destination = np.zeros((dst_height, dst_width), dtype=np.float32)

        reproject(
            source=rasterio.band(src, 1),
            destination=destination,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=ref_profile['transform'],
            dst_crs=ref_profile['crs'],
            resampling=Resampling.bilinear,
            dst_nodata=np.nan
        )
        return destination


# ---------------------------------------------------------
# [시각화 1] Before 이미지
# ---------------------------------------------------------
def save_before_plot(data, quantiles, zone_stats, title, save_path):
    mask = (data <= 0) | (np.isnan(data))
    masked_data = np.ma.masked_where(mask, data)

    bounds = [-np.inf] + list(quantiles) + [np.inf]
    cmap = colors.ListedColormap(COLOR_LIST)
    norm = colors.BoundaryNorm(bounds, cmap.N)
    cmap.set_bad(color='white')

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(masked_data, cmap=cmap, norm=norm, interpolation='nearest')
    ax.axis('off')
    ax.set_title(title, fontsize=15, pad=20)

    legend_patches = []
    for i, stats in enumerate(zone_stats):
        z_name = ZONE_NAMES[i]
        if stats['count'] > 0:
            label_text = (f"{z_name}\n"
                          f"  Range: {stats['min']:.3f} ~ {stats['max']:.3f}\n"
                          f"  Avg: {stats['mean']:.3f}")
        else:
            label_text = f"{z_name}: No Data"
        patch = mpatches.Patch(color=COLOR_LIST[i], label=label_text)
        legend_patches.append(patch)

    plt.legend(handles=legend_patches, bbox_to_anchor=(1.05, 1), loc='upper left',
               title="Before Zones Info", fontsize=9)
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1, dpi=150)
    plt.close()
    print(f"      └ [이미지 1] {os.path.basename(save_path)}")


# ---------------------------------------------------------
# [시각화 2] After 이미지 (Zone별)
# ---------------------------------------------------------
def save_after_zone_plot(zone_map, data_a_aligned, mean_b_list, title, save_path):
    cmap_zones = colors.ListedColormap(['white'] + COLOR_LIST)
    bounds_zones = [0, 1, 2, 3, 4, 5, 6]
    norm_zones = colors.BoundaryNorm(bounds_zones, cmap_zones.N)

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(zone_map, cmap=cmap_zones, norm=norm_zones, interpolation='nearest')
    ax.axis('off')
    ax.set_title(title, fontsize=15, pad=20)

    legend_patches = []
    for z_id in range(1, 6):
        mask = (zone_map == z_id)
        vals_a = data_a_aligned[mask & (data_a_aligned > 0) & (~np.isnan(data_a_aligned))]

        if len(vals_a) > 0:
            a_min = np.min(vals_a)
            a_max = np.max(vals_a)
            a_mean = np.mean(vals_a)
            b_mean = mean_b_list[z_id - 1]
            growth_rate = (a_mean - b_mean) / b_mean * 100 if b_mean != 0 else 0.0

            label_text = (f"{ZONE_NAMES[z_id - 1]}\n"
                          f"  Range: {a_min:.3f}~{a_max:.3f}\n"
                          f"  Avg: {a_mean:.3f} ({growth_rate:+.1f}%)")
        else:
            label_text = f"{ZONE_NAMES[z_id - 1]}: No Data"
        patch = mpatches.Patch(color=COLOR_LIST[z_id - 1], label=label_text)
        legend_patches.append(patch)

    plt.legend(handles=legend_patches, bbox_to_anchor=(1.05, 1), loc='upper left',
               title="After Stats by Before Zones", fontsize=9)
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1, dpi=150)
    plt.close()
    print(f"      └ [이미지 2] {os.path.basename(save_path)}")


# ---------------------------------------------------------
# [시각화 3] 성장률 지도
# ---------------------------------------------------------
def save_growth_plot(growth_map, title, save_path):
    mask = np.isnan(growth_map) | np.isinf(growth_map)
    masked_data = np.ma.masked_where(mask, growth_map)

    max_val = 100.0
    cmap = plt.cm.RdBu
    norm = colors.Normalize(vmin=-max_val, vmax=max_val)
    cmap.set_bad(color='white')

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(masked_data, cmap=cmap, norm=norm, interpolation='nearest')
    ax.axis('off')
    ax.set_title(title, fontsize=15, pad=20)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Growth Rate (%)', rotation=270, labelpad=15)
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1, dpi=150)
    plt.close()
    print(f"      └ [이미지 3] {os.path.basename(save_path)}")


# ---------------------------------------------------------
# [시각화 4] 막대그래프 (신규 추가)
# ---------------------------------------------------------
def save_growth_bar_chart(stats_summary, field_code, save_path):
    """
    Zone별 Before/After Mean 비교 막대그래프 생성
    """
    zones = [f"Zone {i}" for i in range(1, 6)]
    before_means = [d['before'] for d in stats_summary]
    after_means = [d['after'] for d in stats_summary]
    rates = [d['rate'] for d in stats_summary]

    x = np.arange(len(zones))  # 레이블 위치
    width = 0.35  # 막대 너비

    fig, ax = plt.subplots(figsize=(10, 6))

    # 막대 생성 (Before: 회색, After: 파란색)
    rects1 = ax.bar(x - width / 2, before_means, width, label='Before', color='#aaaaaa', alpha=0.8)
    rects2 = ax.bar(x + width / 2, after_means, width, label='After', color='#1f77b4', alpha=0.9)

    # 레이블링
    ax.set_ylabel('GNDVI Mean')
    ax.set_title(f'{field_code} Growth Comparison by Zone')
    ax.set_xticks(x)
    ax.set_xticklabels(zones)
    ax.legend()

    # Y축 범위 설정 (화살표 공간 확보)
    max_val = max(max(before_means), max(after_means)) if before_means else 0
    ax.set_ylim(0, max_val * 1.25)

    # 증감율 텍스트/화살표 표시
    for i in range(len(zones)):
        h_a = after_means[i]
        h_b = before_means[i]
        rate = rates[i]

        if h_b == 0: continue  # 데이터 없는 경우 패스

        # 표시 위치 (더 높은 막대 위)
        y_pos = max(h_a, h_b) + (max_val * 0.02)

        # 스타일 결정
        if rate > 0:
            marker = "▲"
            color = "#d62728"  # 빨강 (성장)
            text = f"+{rate:.1f}%"
        elif rate < 0:
            marker = "▼"
            color = "#1f77b4"  # 파랑 (감소)
            text = f"{rate:.1f}%"
        else:
            marker = "-"
            color = "black"
            text = "0.0%"

        # 텍스트 추가
        ax.text(x[i], y_pos, f"{marker}\n{text}", ha='center', va='bottom',
                color=color, fontweight='bold', fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"      └ [이미지 4] {os.path.basename(save_path)}")


def process_field(field_code):
    print(f"\n--- [{field_code}] 분석 시작 ---")

    path_b = find_file(DIR_BEFORE, field_code)
    path_a = find_file(DIR_AFTER, field_code)
    if not path_b or not path_a:
        print("[오류] 파일 누락")
        return None

    # 1. Before 로드
    data_b, mask_b, prof_b = read_and_resample(path_b, target_gsd=TARGET_GSD)
    if np.sum(mask_b) == 0: return None

    # 2. Zone 설정
    valid_b = data_b[mask_b]
    quantiles = np.percentile(valid_b, [20, 40, 60, 80])
    bins = [-np.inf] + list(quantiles) + [np.inf]
    zone_map = np.zeros_like(data_b, dtype=np.uint8)
    zone_map[mask_b] = np.digitize(data_b[mask_b], bins)

    # Before Stats 계산
    before_stats_list = []
    mean_b_list = []
    for z in range(1, 6):
        vals = data_b[(zone_map == z)]
        if len(vals) > 0:
            stats = {'min': np.min(vals), 'max': np.max(vals), 'mean': np.mean(vals), 'count': len(vals)}
        else:
            stats = {'min': 0, 'max': 0, 'mean': 0, 'count': 0}
        before_stats_list.append(stats)
        mean_b_list.append(stats['mean'])

    # 3. After 정합
    data_a_aligned = align_image(path_a, prof_b)

    # 4. 성장률 맵
    with np.errstate(divide='ignore', invalid='ignore'):
        growth_map = (data_a_aligned - data_b) / data_b * 100
        growth_map[~mask_b] = np.nan
        growth_map[data_a_aligned <= 0] = np.nan

    # ==========================
    # 이미지 저장
    # ==========================
    # [1] Before
    save_before_plot(data_b, quantiles, before_stats_list,
                     f"{field_code} Before Status",
                     os.path.join(DIR_OUTPUT, f"{field_code}_01_Before.png"))
    # [2] After Zone
    save_after_zone_plot(zone_map, data_a_aligned, mean_b_list,
                         f"{field_code} After Status (by Zones)",
                         os.path.join(DIR_OUTPUT, f"{field_code}_02_After_ZoneStats.png"))
    # [3] Growth Map
    save_growth_plot(growth_map, f"{field_code} Growth Rate (%)",
                     os.path.join(DIR_OUTPUT, f"{field_code}_03_GrowthRate.png"))

    # ==========================
    # 데이터 집계 및 차트 생성
    # ==========================
    results = []
    stats_summary = []  # 차트용 데이터

    for z in range(1, 6):
        z_mask = (zone_map == z)
        overlap = z_mask & (data_a_aligned > 0) & (~np.isnan(data_a_aligned))

        # 기본값
        mean_b = mean_b_list[z - 1]
        mean_a = 0.0
        diff = 0.0
        rate = 0.0
        a_min, a_max = 0.0, 0.0
        area = 0

        if np.sum(overlap) > 0:
            vals_a = data_a_aligned[overlap]
            vals_b = data_b[overlap]  # 위치 매칭

            mean_a = np.mean(vals_a)
            mean_b_overlap = np.mean(vals_b)  # 겹치는 부분의 Before 평균 (전체 평균과 미세하게 다를 수 있음)

            # 차트/CSV의 일관성을 위해 겹치는 영역 기준으로 다시 계산하는 것이 더 정확할 수 있으나
            # 여기서는 전체 Zone 평균(mean_b)과 비교하겠습니다.

            diff = mean_a - mean_b
            rate = (diff / mean_b * 100) if mean_b != 0 else 0.0

            a_min, a_max = np.min(vals_a), np.max(vals_a)
            area = np.sum(overlap)

        # CSV 데이터
        results.append({
            'Field': field_code,
            'Zone': f"Zone {z}",
            'Area_m2': area,
            'Before_Min': round(before_stats_list[z - 1]['min'], 4),
            'Before_Mean': round(mean_b, 4),
            'Before_Max': round(before_stats_list[z - 1]['max'], 4),
            'After_Min': round(a_min, 4),
            'After_Mean': round(mean_a, 4),
            'After_Max': round(a_max, 4),
            'Change_Value': round(diff, 4),
            'Growth_Rate(%)': round(rate, 2)
        })

        # 차트용 데이터
        stats_summary.append({
            'before': mean_b,
            'after': mean_a,
            'rate': rate
        })

    # [4] 막대 그래프 저장
    save_growth_bar_chart(stats_summary, field_code,
                          os.path.join(DIR_OUTPUT, f"{field_code}_04_GrowthChart.png"))

    return results


# =========================================================
# 메인 실행
# =========================================================
if __name__ == "__main__":
    if not os.path.exists(DIR_OUTPUT):
        os.makedirs(DIR_OUTPUT)

    print(f">>> 작업 시작 (Target GSD: {TARGET_GSD}m)")
    codes = get_field_codes(DIR_BEFORE)
    all_data = []

    for c in codes:
        try:
            res = process_field(c)
            if res: all_data.extend(res)
        except Exception as e:
            print(f"[오류] {c}: {e}")
            import traceback

            traceback.print_exc()

    if all_data:
        df = pd.DataFrame(all_data)
        csv_path = os.path.join(DIR_OUTPUT, "Zone_Growth_Analysis_Final.csv")
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"\n>>> 완료. 결과 저장됨: {csv_path}")
