import numpy as np
from scipy.optimize import curve_fit
from skimage.morphology import skeletonize, binary_dilation
import scipy.ndimage as ndi
from scipy.ndimage import generate_binary_structure
from skimage.measure import label
from scipy.optimize import curve_fit


def SkeletalSimilarity(SrcVessels, RefVessels, Mask=None, Alpha=0.5, Levels=2):
    batch_size = SrcVessels.shape[0]
    rSe_batch, rSp_batch, rAcc_batch, SS_batch, Confidence_batch = [], [], [], [], []
    for i in range(batch_size):
        src_vessels = SrcVessels[i]
        ref_vessels = RefVessels[i]
        mask = np.ones_like(src_vessels)
        rSe, rSp, rAcc, SS, Confidence, _ = SkeletalSimilaritySingle(src_vessels, ref_vessels, mask, Alpha, Levels)
        rSe_batch.append(rSe)
        rSp_batch.append(rSp)
        rAcc_batch.append(rAcc)
        SS_batch.append(SS)
        Confidence_batch.append(Confidence)
    return np.mean(rSe_batch), np.mean(rSp_batch), np.mean(rAcc_batch), np.mean(SS_batch)

def SkeletalSimilaritySingle(SrcVessels, RefVessels, Mask=None, Alpha=0.5, Levels=2):
    minLength = 4  # the predefined minimum length of the skeleton segment
    maxLength = 15  # the predefined maximum length of the skeleton segment
    Mask = np.ones_like(SrcVessels)

    # Initialization
    Mask[Mask > 0] = 1
    height, width = RefVessels.shape
    SrcVessels = np.uint8(SrcVessels)
    SrcVessels[SrcVessels > 0] = 1
    SrcSkeleton = skeletonize(SrcVessels)
    RefVessels = np.uint8(RefVessels)
    RefVessels[RefVessels > 0] = 1
    RefSkeleton = skeletonize(RefVessels)

    # Generate the searching range of each pixel
    # RefThickness, RefminRadius, RefmaxRadius = CalcThickness(RefSkeleton, RefVessels)
    RefThickness, RefminRadius, RefmaxRadius = calc_thickness(RefSkeleton, RefVessels)


    bin = (RefmaxRadius - RefminRadius) * 1.0 / Levels
    SearchingRadius = np.ceil((RefmaxRadius - RefThickness + 0.0001) * 1.0 / bin)
    SearchingRadius = np.minimum(SearchingRadius, Levels)
    SearchingRadius[RefSkeleton == 0] = 0

    # Calc the vessel thickness of each pixel in Src
    SrcThickness, SrcminRadius, SrcmaxRadius = calc_thickness(SrcSkeleton, SrcVessels)

    SearchingMask = generate_range(SearchingRadius, Mask)
    # SearchingMask = GenerateRange(SearchingRadius, Mask)

    # Delete wrong skeleton segments
    SrcSkeleton[SearchingMask == 0] = 0

    # Segment the target skeleton map

    SegmentID = SegmentSkeleton(RefSkeleton, minLength, maxLength)
    SegmentID[Mask == 0] = 0
    # Calculate the confidence
    OriginalSkeleton = RefSkeleton
    EvaluationSkeleton = SegmentID.copy()
    EvaluationSkeleton[EvaluationSkeleton > 0] = 1
    Confidence = np.sum(EvaluationSkeleton) * 1.0 / np.sum(OriginalSkeleton)

    # Calculate the skeletal similarity for each segment
    SS = 0.0
    for Index in range(1, np.max(SegmentID) + 1):
        SegmentRadius = SearchingRadius.copy()
        SegmentRadius[SegmentID != Index] = 0
        SegmentMask = generate_range(SegmentRadius, Mask)
        SrcSegment = SrcSkeleton.copy()
        SrcSegment[SegmentMask == 0] = 0

        # Remove additionally selected pixels
        SrcSegment = NoiseRemoval(SrcSegment, RefSkeleton, SegmentID, Index)
        SrcX, SrcY = np.where(SrcSegment > 0)
        RefX, RefY = np.where(SegmentID == Index)

        # Calc average vessel thickness of Src skeleton
        SkeletonTemp = SrcSkeleton.copy()
        SkeletonTemp[SrcSegment == 0] = 0
        SrcAvgThickness = np.sum(SrcThickness * SkeletonTemp) / len(SrcX)
        # Calc average vessel thickness of Ref skeleton
        SkeletonTemp = RefSkeleton.copy()
        SkeletonTemp[SegmentID != Index] = 0
        RefAvgThickness = np.sum(RefThickness * SkeletonTemp) / len(RefX)

        RefAvgRange = np.sum(SegmentRadius) * 1.0 / len(RefX)

        if len(np.unique(SrcX)) > len(np.unique(SrcY)):
            SS = SS + CalcSimilarity(SrcX, SrcY, SrcAvgThickness, RefX, RefY, RefAvgThickness, Alpha,
                                     RefAvgRange) * len(RefX)
        else:
            SS = SS + CalcSimilarity(SrcY, SrcX, SrcAvgThickness, RefY, RefX, RefAvgThickness, Alpha,
                                     RefAvgRange) * len(RefX)

    SegmentID[SegmentID > 0] = 1
    SS = SS / np.sum(SegmentID)

    PositiveMask = SearchingMask + RefVessels
    PositiveMask[PositiveMask > 0] = 1
    PositiveMask[Mask == 0] = 0
    TP = SS * np.sum(PositiveMask)
    FN = (1 - SS) * np.sum(PositiveMask)
    FP = np.sum(SrcVessels * (1 - PositiveMask) * Mask)
    TN = np.sum((1 - SrcVessels) * (1 - PositiveMask) * Mask)
    rSe = TP * 100.0 / (TP + FN)
    rSp = TN * 100.0 / (TN + FP)
    rAcc = (TP + TN) * 100 / (TP + FN + TN + FP)

    return rSe, rSp, rAcc, SS, Confidence, SearchingMask


def poly3(x, a, b, c, d):
    return a * x ** 3 + b * x ** 2 + c * x + d

def CalcSimilarity(SrcX, SrcY, SrcAvgThickness, RefX, RefY, RefAvgThickness, Alpha, RefAvgRange):
    Score = 0.0

    RefX = np.array(RefX)
    unique_values, counts = np.unique(RefX, return_counts=True)
    duplicates = unique_values[counts > 1]

    # 生成一个新的唯一数组
    unique_array = np.concatenate((unique_values[counts == 1], duplicates))
    for duplicate in duplicates:
        duplicate_indices = np.where(RefX == duplicate)[0]
        new_values = duplicate + np.arange(len(duplicate_indices)-1) * 0.01 + np.random.rand(
            len(duplicate_indices)-1) * 1e-6
        unique_array = np.insert(unique_array.astype(np.float64), len(unique_array)-1, new_values)


    # 将RefX替换为唯一数组
    RefX = unique_array
    # 拟合参考骨架的多项式
    RefPoly, _ = curve_fit(poly3, RefX, RefY)

    if len(SrcX) > 0.6 * len(RefX) and len(SrcX) > 3:
        # 确保SrcX中的值是唯一的
        SrcX = np.array(SrcX)
        unique_values_SrcX, counts_SrcX = np.unique(SrcX, return_counts=True)
        duplicates_SrcX = unique_values_SrcX[counts_SrcX > 1]

        # 生成一个新的唯一数组
        unique_array_SrcX = np.concatenate((unique_values_SrcX[counts_SrcX == 1], duplicates_SrcX))
        for duplicate in duplicates_SrcX:
            duplicate_indices = np.where(SrcX == duplicate)[0]
            new_values = duplicate + np.arange(len(duplicate_indices)-1) * 0.01 + np.random.rand(
                len(duplicate_indices)-1) * 1e-6
            unique_array_SrcX = np.insert(unique_array_SrcX.astype(np.float64), len(unique_array_SrcX)-1, new_values)

        # 将SrcX替换为唯一数组
        SrcX = unique_array_SrcX

        # 拟合源骨架的多项式
        SrcPoly, _ = curve_fit(poly3, SrcX, SrcY)

        # 计算相似度
        Similarity = np.abs(np.dot(SrcPoly, RefPoly) / (np.linalg.norm(SrcPoly) + 1e-10) / (np.linalg.norm(RefPoly) + 1e-10))

        # 计算厚度差异
        Thickness = 1.0 - np.abs(RefAvgThickness - SrcAvgThickness) * 1.0 / RefAvgRange
        Thickness = max(Thickness, 0)

        # 计算总分数
        Score = (1 - Alpha) * Similarity + Alpha * Thickness

    return Score


def NoiseRemoval(SrcSegment, RefSkeleton, SegmentID, ID):
    height, width = SegmentID.shape
    UpdatedSegment = SrcSegment.copy()
    X, Y = np.where(SrcSegment > 0)
    for Index in range(len(X)):
        minRadius = 10
        minID = 0
        if SegmentID[X[Index], Y[Index]] > 0:
            if SegmentID[X[Index], Y[Index]] != ID:
                UpdatedSegment[X[Index], Y[Index]] = 0
            continue
        else:
            for x in range(max(X[Index] - 5, 0), min(X[Index] + 6, height)):
                for y in range(max(Y[Index] - 5, 0), min(Y[Index] + 6, width)):
                    if x == X[Index] and y == Y[Index]:
                        continue
                    if RefSkeleton[x, y] > 0:
                        if np.sqrt((x - X[Index]) ** 2 + (y - Y[Index]) ** 2) < minRadius or (
                                np.sqrt((x - X[Index]) ** 2 + (y - Y[Index]) ** 2) == minRadius and SegmentID[
                            x, y] == ID):
                            minID = SegmentID[x, y]
                            minRadius = np.sqrt((x - X[Index]) ** 2 + (y - Y[Index]) ** 2)
        if minID != ID:
            UpdatedSegment[X[Index], Y[Index]] = 0

    return UpdatedSegment


def calc_thickness(skeleton, guidance):
    thickness = np.zeros_like(skeleton, dtype=np.float64)
    binary_guidance = guidance.astype(bool).astype(int)
    distmap = ndi.distance_transform_edt(binary_guidance)
    thickness[skeleton > 0] = distmap[skeleton > 0]
    min_radius = np.min(thickness[skeleton > 0])
    max_radius = np.max(thickness[skeleton > 0])
    return thickness, min_radius, max_radius





def generate_range(SearchingRadius, Mask):
    height, width = SearchingRadius.shape

    # 生成坐标矩阵
    x, y = np.meshgrid(np.arange(width), np.arange(height), indexing='ij')

    # 找出SearchingRadius大于0的位置
    valid_mask = SearchingRadius > 0
    valid_x, valid_y = np.where(valid_mask)
    valid_radius = SearchingRadius[valid_mask]

    # 初始化SearchingMask
    SearchingMask = np.zeros_like(SearchingRadius, dtype=np.uint8)

    # 对于每个有效的骨架点
    for i in range(len(valid_x)):
        x_i, y_i = valid_x[i], valid_y[i]
        radius = valid_radius[i]

        # 计算与该点的距离
        dx = np.abs(x - x_i)
        dy = np.abs(y - y_i)
        dist = np.maximum(dx, dy)

        # 限制搜索范围在边长为20的正方形内
        x_min, x_max = max(0, x_i - 10), min(height - 1, x_i + 10)
        y_min, y_max = max(0, y_i - 10), min(width - 1, y_i + 10)

        # 标记距离小于等于半径的点
        SearchingMask[x_min:x_max + 1, y_min:y_max + 1][dist[x_min:x_max + 1, y_min:y_max + 1] <= radius] = 1

    # 应用Mask
    SearchingMask *= Mask

    return SearchingMask



from skimage.measure import label
def SegmentSkeleton(RefSkeleton, minLength, maxLength):
    height, width = RefSkeleton.shape
    RefSkeleton = np.uint8(RefSkeleton)

    X, Y = np.where(RefSkeleton > 0)
    IntersectingPixel = RefSkeleton.copy()
    for Index in range(len(X)):
        top = max(X[Index] - 1, 0)
        bottom = min(X[Index] + 1, height - 1)
        left = max(Y[Index] - 1, 0)
        right = min(Y[Index] + 1, width - 1)
        if np.sum(RefSkeleton[top:bottom + 1, left:right + 1]) > 3:
            RefSkeleton[X[Index], Y[Index]] = 0
    IntersectingPixel = IntersectingPixel - RefSkeleton
    # Delete segments smaller than minLength
    L, num = label(RefSkeleton, connectivity=2, return_num=True)
    for Index in range(1, num + 1):
        Component = RefSkeleton.copy()
        Component[L != Index] = 0
        Component[Component > 0] = 1
        if np.sum(Component) < minLength:
            RefSkeleton[L == Index] = 0

    L, num = label(RefSkeleton, connectivity=2, return_num=True)
    SegmentID = L
    # Cut segments longer than maxLength
    for Index in range(1, num + 1):
        Component = SegmentID.copy()
        Component[Component != Index] = 0
        Component[Component > 0] = 1
        L = np.sum(Component)
        if L > maxLength:
            SegmentID[SegmentID == Index] = 0
            UpdateSegmentID = CutSegment(Component, L, Index, np.max(SegmentID), maxLength)
            SegmentID = SegmentID + UpdateSegmentID
    # Assign ID to intersecting pixels
    X, Y = np.where(IntersectingPixel > 0)
    for Index in range(len(X)):
        top = max(X[Index] - 1, 0)
        bottom = min(X[Index] + 1, height - 1)
        left = max(Y[Index] - 1, 0)
        right = min(Y[Index] + 1, width - 1)
        IDs = np.unique(SegmentID[top:bottom + 1, left:right + 1])
        MinimumLength = 100
        ID = 0
        for index in range(1, len(IDs)):
            if len(np.where(SegmentID == IDs[index])[0]) < MinimumLength:
                ID = IDs[index]
        SegmentID[X[Index], Y[Index]] = ID

    return SegmentID


def CutSegment(Segment, L, Index, ID, maxLength):
    height, width = Segment.shape
    UpdateSegmentID = Segment.copy()
    nums = int(np.floor(L / maxLength))
    TarLength = round(L / nums)
    X, Y = np.where(Segment > 0)
    for index in range(len(X)):
        top = max(X[index] - 1, 0)
        bottom = min(X[index] + 1, height - 1)
        left = max(Y[index] - 1, 0)
        right = min(Y[index] + 1, width - 1)
        if np.sum(Segment[top:bottom + 1, left:right + 1]) == 2:
            startX = X[index]
            startY = Y[index]
            break
    IDs = np.zeros(len(X), dtype=np.uint32)
    SubLength = 1
    count = 1
    while count < nums:
        IDs[index] = count
        for index in range(len(X)):
            distance = np.sqrt((X[index] - startX) ** 2 + (Y[index] - startY) ** 2)
            if np.floor(distance) == 1 and IDs[index] == 0:
                startX = X[index]
                startY = Y[index]
                if SubLength < TarLength:
                    SubLength = SubLength + 1
                else:
                    SubLength = 1
                    count = count + 1
                break
    for index in range(len(X)):
        if IDs[index] > 0:
            UpdateSegmentID[X[index], Y[index]] = IDs[index] + ID
        else:
            UpdateSegmentID[X[index], Y[index]] = Index

    return UpdateSegmentID


if __name__ == '__main__':
    from PIL import Image
    import numpy as np
    import time
    #time start
    start = time.time()
    SrcVessels = np.array(Image.open("/data/xiaoyi/Fundus/STARE/train/mask_512_png/im0001.png"))
    RefVessels = SrcVessels.copy()
    SrcVessels = np.repeat(SrcVessels[np.newaxis, :, :], 3, axis=0)
    RefVessels = np.repeat(RefVessels[np.newaxis, :, :], 3, axis=0)

    Mask = np.ones_like(SrcVessels)
    Alpha = 0.5
    Levels = 2
    SkeletalSimilarity(SrcVessels, RefVessels, Mask, Alpha, Levels)
    #time end
    end = time.time()
    print('time:',end-start)