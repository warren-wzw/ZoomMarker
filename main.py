import os
import cv2
import numpy as np

# ========== 配置参数 ==========
colors = [(255, 0, 0), (0, 255, 0)]  # 红绿交替颜色
border_thickness = 2
min_selection_size = 10
input_root = "./images"
output_root = "./out"

# ========== 全局状态 ==========
regions = []           # 存储Region对象的列表
current_region = None  # 当前临时选区（字典）
current_dragging = -1  # 当前拖动区域索引
dragging = False       # 拖动状态标志
original_img = None    # 原始图像
img_name = ""          # 当前处理的图片名称

class Region:
    def __init__(self, start, end, color, zoom_factor):
        # 规范坐标顺序
        self.start = (min(start[0], end[0]), min(start[1], end[1]))
        self.end = (max(start[0], end[0]), max(start[1], end[1]))
        self.color = color
        self.zoom_factor = max(1.0, float(zoom_factor))
        self.zoomed_img = None
        self.pos = None
        self._create_zoom()

    def _create_zoom(self):
        """生成放大区域"""
        try:
            x1, y1 = self.start
            x2, y2 = self.end
            roi = original_img[y1:y2, x1:x2]
            self.zoomed_img = cv2.resize(
                roi, 
                (int((x2-x1)*self.zoom_factor), int((y2-y1)*self.zoom_factor)),
                interpolation=cv2.INTER_CUBIC
            )
            # 初始位置到选区右侧
            img_h, img_w = original_img.shape[:2]
            self.pos = (
                min(x2 + 10, img_w - self.zoomed_img.shape[1]),
                min(y1, img_h - self.zoomed_img.shape[0])
            )
        except Exception as e:
            print(f"放大失败: {str(e)}")

class RegionData:
    """选区元数据容器"""
    def __init__(self):
        self.ref_width = 0
        self.ref_height = 0
        self.regions = []  # 格式: (start, end, color, zoom_factor, pos)
    
    def add_region(self, region):
        """添加区域数据"""
        self.regions.append((
            region.start, 
            region.end, 
            region.color,
            region.zoom_factor,
            region.pos
        ))

def get_zoom_factor():
    """获取用户输入的放大倍数"""
    while True:
        try:
            factor = input("请输入放大倍数（默认2.0）: ")
            return max(1.0, float(factor)) if factor else 2.0
        except:
            print("⚠️ 输入无效，请填写数字")

def process_single_image(img_path, save_path):
    """处理单个图像文件"""
    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    if img is None: return
    
    h, w = img.shape[:2]
    output = img.copy()
    
    for (start, end, color, factor, ref_pos) in region_data.regions:
        # 计算缩放比例
        scale_x = w / region_data.ref_width
        scale_y = h / region_data.ref_height
        
        # 缩放原始选区坐标
        x1 = int(start[0] * scale_x)
        y1 = int(start[1] * scale_y)
        x2 = int(end[0] * scale_x)
        y2 = int(end[1] * scale_y)
        
        # 缩放位置坐标
        pos_x = int(ref_pos[0] * scale_x)
        pos_y = int(ref_pos[1] * scale_y)
        
        # 边界保护
        pos_x = max(0, min(pos_x, w - int((x2-x1)*factor)))
        pos_y = max(0, min(pos_y, h - int((y2-y1)*factor)))
        
        # 绘制原始框
        cv2.rectangle(output, (x1,y1), (x2,y2), color, border_thickness)
        
        # 生成放大区域
        try:
            roi = img[y1:y2, x1:x2]
            zoomed = cv2.resize(
                roi, 
                (int((x2-x1)*factor), int((y2-y1)*factor)),
                interpolation=cv2.INTER_CUBIC
            )
            
            # 覆盖放大区域
            output[pos_y:pos_y+zoomed.shape[0], pos_x:pos_x+zoomed.shape[1]] = zoomed
            cv2.rectangle(output, 
                        (pos_x, pos_y),
                        (pos_x+zoomed.shape[1], pos_y+zoomed.shape[0]),
                        color, border_thickness)
        except Exception as e:
            print(f"处理失败: {str(e)}")
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, cv2.cvtColor(output, cv2.COLOR_RGB2BGR))

def batch_process():
    """批量处理所有同名图片"""
    print("🔍 开始批量处理...")
    for root, dirs, files in os.walk(input_root):
        for file in files:
            if file == img_name:
                in_path = os.path.join(root, file)
                rel_path = os.path.relpath(root, input_root)
                out_dir = os.path.join(output_root, rel_path)
                out_path = os.path.join(out_dir, file)
                process_single_image(in_path, out_path)
    print("✅ 批量处理完成")

def mouse_callback(event, x, y, flags, param):
    global current_region, current_dragging, dragging
    
    # 左键按下 - 开始新选区
    if event == cv2.EVENT_LBUTTONDOWN:
        color = colors[len(regions) % 2]
        current_region = {'start': (x, y), 'end': (x, y), 'color': color}
    
    # 鼠标移动 - 更新临时选区
    elif event == cv2.EVENT_MOUSEMOVE:
        if current_region is not None:
            current_region['end'] = (x, y)
            update_display()
    
    # 左键释放 - 结束绘制并获取倍数
    elif event == cv2.EVENT_LBUTTONUP:
        if current_region is not None:
            start = current_region['start']
            end = current_region['end']
            color = current_region['color']
            
            # 检查选区有效性
            if abs(end[0]-start[0]) < min_selection_size or abs(end[1]-start[1]) < min_selection_size:
                print("⚠️ 选区过小，已忽略")
                current_region = None
                return
            
            # 获取放大倍数
            print(f"当前颜色：{'红色' if color==(255,0,0) else '绿色'}")
            zoom_factor = get_zoom_factor()
            
            # 创建区域对象
            try:
                new_region = Region(start, end, color, zoom_factor)
                if new_region.zoomed_img is not None:
                    regions.append(new_region)
                    print(f"✅ 区域{len(regions)}已创建（{zoom_factor}倍）")
            except Exception as e:
                print(f"区域创建失败: {str(e)}")
            
            current_region = None
            update_display()
    
    # 中键按下 - 开始拖动
    elif event == cv2.EVENT_MBUTTONDOWN:
        for i, r in enumerate(regions):
            if r.pos and r.zoomed_img is not None:
                rx, ry = r.pos
                rw, rh = r.zoomed_img.shape[1], r.zoomed_img.shape[0]
                if rx <= x <= rx+rw and ry <= y <= ry+rh:
                    current_dragging = i
                    dragging = True
                    break
    
    # 中键释放 - 停止拖动
    elif event == cv2.EVENT_MBUTTONUP:
        dragging = False
        current_dragging = -1
    
    # 拖动处理
    if dragging and current_dragging != -1:
        region = regions[current_dragging]
        img_h, img_w = original_img.shape[:2]
        new_x = x - region.zoomed_img.shape[1] // 2
        new_y = y - region.zoomed_img.shape[0] // 2
        new_x = max(0, min(new_x, img_w - region.zoomed_img.shape[1]))
        new_y = max(0, min(new_y, img_h - region.zoomed_img.shape[0]))
        region.pos = (new_x, new_y)
        update_display()

def update_display():
    """更新界面显示"""
    display = original_img.copy()
    
    # 绘制已确认区域
    for r in regions:
        cv2.rectangle(display, r.start, r.end, r.color, border_thickness)
        if r.zoomed_img is not None and r.pos:
            x, y = r.pos
            h, w = r.zoomed_img.shape[:2]
            overlay = display.copy()
            overlay[y:y+h, x:x+w] = r.zoomed_img
            cv2.rectangle(overlay, (x, y), (x+w, y+h), r.color, border_thickness)
            cv2.addWeighted(overlay, 0.7, display, 0.3, 0, display)
    
    # 绘制临时选区
    if current_region is not None:
        start = current_region['start']
        end = current_region['end']
        color = current_region['color']
        cv2.rectangle(display, start, end, color, border_thickness)
    
    cv2.imshow("Image Processor", display)

if __name__ == "__main__":
    # 初始化参考图像
    demo_path = os.path.join(input_root, "Ours/01380.png")
    original_img = cv2.cvtColor(cv2.imread(demo_path), cv2.COLOR_BGR2RGB)
    if original_img is None:
        raise FileNotFoundError("无法加载参考图像")
    
    img_name = os.path.basename(demo_path)
    region_data = RegionData()
    region_data.ref_width = original_img.shape[1]
    region_data.ref_height = original_img.shape[0]
    
    # 创建窗口
    cv2.namedWindow("Image Processor")
    cv2.setMouseCallback("Image Processor", mouse_callback)
    update_display()
    
    print("操作指南：")
    print("1. 左键拖动创建选区 → 自动弹出倍数输入")
    print("2. 中键拖动调整放大区域位置")
    print("3. 按 S 保存并批量处理")
    print("4. 按 Q 退出")
    
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            # 保存元数据
            region_data.regions = []
            for r in regions:
                region_data.add_region(r)
            
            # 处理并保存
            process_single_image(demo_path, os.path.join(output_root, img_name))
            batch_process()
            print("✅ 已保存所有修改")
    
    cv2.destroyAllWindows()



