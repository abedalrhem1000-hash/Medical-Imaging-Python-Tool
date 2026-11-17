import sys
import subprocess

# ==========================================
# 0. فحص وتثبيت المكتبات المطلوبة تلقائياً
# ==========================================
required_libraries = {
    'numpy': 'numpy',
    'nibabel': 'nibabel',
    'pydicom': 'pydicom',
    'matplotlib': 'matplotlib',
    'scipy': 'scipy'
}

missing_libs = []
for lib_name, pip_name in required_libraries.items():
    try:
        __import__(lib_name)
    except ImportError:
        missing_libs.append(pip_name)

if missing_libs:
    import tkinter as tk
    from tkinter import messagebox
    
    root_temp = tk.Tk()
    root_temp.withdraw()
    
    msg = f"المكتبات التالية غير متوفرة على جهازك:\n{', '.join(missing_libs)}\n\nهل تريد تثبيتها تلقائياً الآن للبدء؟"
    answer = messagebox.askyesno("تثبيت المكتبات المطلوبة - Medical Workstation", msg)
    
    if answer:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing_libs)
        except Exception as e:
            messagebox.showerror("خطأ في التثبيت", f"حدث خطأ أثناء تثبيت المكتبات:\n{e}")
            sys.exit(1)
    else:
        sys.exit(0)
    root_temp.destroy()

# ==========================================
# استيراد المكتبات بعد التأكد من توفرها
# ==========================================
import os
import tkinter as tk
from tkinter import filedialog
import numpy as np
import nibabel as nib
import pydicom
import matplotlib.pyplot as plt
from matplotlib.widgets import CheckButtons, Slider, Button
from matplotlib.patches import Rectangle
from scipy.ndimage import gaussian_filter, rotate

# إخفاء نافذة تكنسر الرئيسية
root = tk.Tk()
root.withdraw()

# ==========================================
# 1. Universal Medical Image Loader
# ==========================================
def load_medical_image(path):
    if os.path.isdir(path):
        dicom_files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith(('.dcm', '.ima')) or '.' not in f]
        datasets = []
        for f in dicom_files:
            try:
                ds = pydicom.dcmread(f, force=True)
                if hasattr(ds, 'pixel_array'):
                    datasets.append(ds)
            except:
                pass
        if not datasets:
            raise ValueError("No valid DICOM/IMA files found in the directory.")
        try:
            datasets.sort(key=lambda x: int(getattr(x, 'InstanceNumber', 0)))
        except:
            pass
        volume = np.stack([ds.pixel_array.astype(np.float32) for ds in datasets], axis=-1)
        seq_name = getattr(datasets[0], 'SeriesDescription', 'IMA_SERIES').upper()
        return volume, seq_name
        
    elif path.endswith(('.nii', '.nii.gz')):
        img = nib.load(path)
        volume = img.get_fdata().astype(np.float32)
        seq_name = os.path.splitext(os.path.basename(path))[0].upper().replace('.NII', '')
        return volume, seq_name
        
    elif path.endswith(('.dcm', '.ima')) or path.upper().endswith('.IMA') or '.' not in path:
        try:
            ds = pydicom.dcmread(path, force=True)
            if not hasattr(ds, 'pixel_array'):
                raise ValueError("File does not contain pixel data.")
            volume = ds.pixel_array.astype(np.float32)
            if volume.ndim == 2:
                volume = np.expand_dims(volume, axis=-1)
            seq_name = getattr(ds, 'SeriesDescription', 'IMA_SINGLE').upper()
            return volume, seq_name
        except Exception as e:
            raise ValueError(f"Could not read IMA/DICOM file: {e}")
    else:
        raise ValueError("Unsupported file format.")

# تحميل السيكونس الأساسي
default_path = r"C:\Users\ABED-ALRAHEM\Downloads\BT0001\brain_t1ce.nii"
loaded_sequences = {}

if os.path.exists(default_path):
    try:
        vol, s_name = load_medical_image(default_path)
        loaded_sequences[s_name] = vol
    except:
        pass

if not loaded_sequences:
    initial_files = filedialog.askopenfilenames(
        title="Select Medical Images (.ima, DICOM, NIfTI)",
        filetypes=[
            ("Medical Images", "*.nii;*.nii.gz;*.dcm;*.ima"),
            ("IMA Files", "*.ima"),
            ("DICOM Files", "*.dcm"),
            ("NIfTI Files", "*.nii;*.nii.gz"),
            ("All Files", "*.*")
        ]
    )
    if initial_files:
        for f_path in initial_files:
            try:
                vol, s_name = load_medical_image(f_path)
                loaded_sequences[s_name] = vol
            except Exception as e:
                print(f"[ERROR] {e}")

if not loaded_sequences:
    loaded_sequences['DEFAULT_BRAIN'] = np.random.rand(240, 240, 155).astype(np.float32)

seq_names = list(loaded_sequences.keys())
default_seq = seq_names[0]

vol_init = loaded_sequences[default_seq]
if vol_init.ndim == 2:
    vol_init = np.expand_dims(vol_init, axis=-1)
nx, ny, nz = vol_init.shape

# ==========================================
# 2. Workstation State & Layout Management
# ==========================================
panels = {
    'p1': {'seq': default_seq, 'slice': nz // 2, 'rot': 0, 'cx': nx // 2, 'cy': ny // 2, 'cz': nz // 2},
    'p2': {'seq': default_seq, 'slice': nz // 2, 'rot': 0, 'cx': nx // 2, 'cy': ny // 2, 'cz': nz // 2}, 
    'p3': {'seq': default_seq, 'slice': nz // 2, 'rot': 0, 'cx': nx // 2, 'cy': ny // 2, 'cz': nz // 2},
    'p4': {'seq': default_seq, 'slice': nz // 2, 'rot': 0, 'cx': nx // 2, 'cy': ny // 2, 'cz': nz // 2}        
}

global_state = {
    'target_panel': 'p1',       
    'box_active': True,
    'box_size': 25,
    'brightness': 0.0,          
    'contrast': 1.0,            
    'sharpness': 1.0,           
    'layout_mode': '2x2',       
    'maximized_panel': None,    
    'updating_slider': False,
    'explorer_scroll_offset': 0 
}

# ==========================================
# 3. Setup Professional Figure Layout & UI
# ==========================================
fig = plt.figure(figsize=(16, 9), facecolor='#121212')
fig.canvas.manager.set_window_title('Professional MRI Workstation - PACS Interface')

# --- أزرار تبديل تخطيط الشاشات العلوي ---
ax_lay_4 = fig.add_axes([0.22, 0.93, 0.06, 0.04])
btn_lay_4 = Button(ax_lay_4, '2x2 (4)', color='#1b5e20', hovercolor='#2e7d32')
for l in btn_lay_4.ax.texts: l.set_color('white'); l.set_fontsize(7); l.set_fontweight('bold')

ax_lay_2h = fig.add_axes([0.29, 0.93, 0.06, 0.04])
btn_lay_2h = Button(ax_lay_2h, '1x2 (2)', color='#0d47a1', hovercolor='#1565c0')
for l in btn_lay_2h.ax.texts: l.set_color('white'); l.set_fontsize(7); l.set_fontweight('bold')

ax_lay_1 = fig.add_axes([0.36, 0.93, 0.06, 0.04])
btn_lay_1 = Button(ax_lay_1, '1x1 (1)', color='#b71c1c', hovercolor='#c62828')
for l in btn_lay_1.ax.texts: l.set_color('white'); l.set_fontsize(7); l.set_fontweight('bold')

btn_lay_4.on_clicked(lambda e: (global_state.update({'layout_mode': '2x2', 'maximized_panel': None}), update_display()))
btn_lay_2h.on_clicked(lambda e: (global_state.update({'layout_mode': '1x2', 'maximized_panel': None}), update_display()))
btn_lay_1.on_clicked(lambda e: (global_state.update({'layout_mode': '1x1', 'maximized_panel': global_state['target_panel']}), update_display()))

# --- زر إظهار/إخفاء مربع الـ Voxel Box ---
ax_check_box = fig.add_axes([0.02, 0.93, 0.16, 0.04])
check_box = CheckButtons(ax_check_box, ['Voxel Box'], [global_state['box_active']])
check_box.ax.set_facecolor('#1e1e1e')
for l in check_box.ax.texts: l.set_color('#00e5ff'); l.set_fontsize(8); l.set_fontweight('bold')
check_box.on_clicked(lambda label: (global_state.update({'box_active': not global_state['box_active']}), update_display()))

# --- أشرطة التحكم ---
ax_slider = fig.add_axes([0.02, 0.88, 0.16, 0.025])
size_slider = Slider(ax=ax_slider, label='Voxel Size', valmin=5, valmax=80, valinit=global_state['box_size'], valstep=1)
for l in size_slider.ax.texts: l.set_color('#00e5ff'); l.set_fontsize(7); l.set_fontweight('bold')
size_slider.ax.set_facecolor('#1e1e1e')
size_slider.on_changed(lambda val: (global_state.update({'box_size': int(val)}), update_display()))

ax_bri = fig.add_axes([0.02, 0.83, 0.16, 0.02])
slider_bri = Slider(ax=ax_bri, label='Brightness', valmin=-100, valmax=100, valinit=0, valstep=1)
for l in slider_bri.ax.texts: l.set_color('#ffcc00'); l.set_fontsize(6)
slider_bri.ax.set_facecolor('#1a1a1a')
slider_bri.on_changed(lambda val: (global_state.update({'brightness': val}), update_display()))

ax_con = fig.add_axes([0.02, 0.79, 0.16, 0.02])
slider_con = Slider(ax=ax_con, label='Contrast', valmin=0.2, valmax=3.0, valinit=1.0, valstep=0.05)
for l in slider_con.ax.texts: l.set_color('#ffcc00'); l.set_fontsize(6)
slider_con.ax.set_facecolor('#1a1a1a')
slider_con.on_changed(lambda val: (global_state.update({'contrast': val}), update_display()))

ax_sha = fig.add_axes([0.02, 0.75, 0.16, 0.02])
slider_sha = Slider(ax=ax_sha, label='Sharpness', valmin=0.5, valmax=2.5, valinit=1.0, valstep=0.1)
for l in slider_sha.ax.texts: l.set_color('#ffcc00'); l.set_fontsize(6)
slider_sha.ax.set_facecolor('#1a1a1a')
slider_sha.on_changed(lambda val: (global_state.update({'sharpness': val}), update_display()))

# --- زر إعادة الضبط ---
ax_reset = fig.add_axes([0.02, 0.70, 0.16, 0.035])
btn_reset = Button(ax_reset, 'Reset Adjustments', color='#b71c1c', hovercolor='#c62828')
for l in btn_reset.ax.texts: l.set_color('white'); l.set_fontsize(7); l.set_fontweight('bold')
btn_reset.on_clicked(lambda e: (global_state.update({'brightness': 0.0, 'contrast': 1.0, 'sharpness': 1.0}), slider_bri.set_val(0.0), slider_con.set_val(1.0), slider_sha.set_val(1.0), update_display()))

# --- شريط البيانات الثابت ---
ax_meta = fig.add_axes([0.02, 0.45, 0.16, 0.23])
ax_meta.set_facecolor('#181818')
for spine in ax_meta.spines.values(): spine.set_edgecolor('#00bcd4'); spine.set_linewidth(1.5)
ax_meta.set_xticks([]); ax_meta.set_yticks([])
meta_text = ax_meta.text(0.05, 0.5, "", color='#00e5ff', fontsize=9, fontweight='bold', va='center', family='monospace')

# --- زر فتح ملفات متعددة ---
ax_open_btn = fig.add_axes([0.02, 0.38, 0.16, 0.05])
open_button = Button(ax_open_btn, '+ Open & Add Files', color='#1b5e20', hovercolor='#2e7d32')
for lbl in open_button.ax.texts: lbl.set_color('white'); lbl.set_fontsize(8); lbl.set_fontweight('bold')

def open_and_add_files(event):
    global seq_names
    file_paths = filedialog.askopenfilenames(
        title="Select Medical Images (.ima, DICOM, NIfTI)",
        filetypes=[
            ("Medical Images", "*.nii;*.nii.gz;*.dcm;*.ima"),
            ("IMA Files", "*.ima"),
            ("DICOM Files", "*.dcm"),
            ("NIfTI Files", "*.nii;*.nii.gz"),
            ("All Files", "*.*")
        ]
    )
    if file_paths:
        added_count = 0
        for file_path in file_paths:
            try:
                vol, s_name = load_medical_image(file_path)
                if s_name in loaded_sequences:
                    s_name = s_name + "_" + str(len(loaded_sequences))
                loaded_sequences[s_name] = vol
                if s_name not in seq_names:
                    seq_names.append(s_name)
                added_count += 1
            except Exception as e:
                print(f"[ERROR] {e}")
        if added_count > 0:
            refresh_explorer_sidebar()
            update_display()

open_button.on_clicked(open_and_add_files)

# --- الشريط الجانبي (DICOM Explorer) مع شريط سكرول ---
ax_explorer = fig.add_axes([0.02, 0.13, 0.16, 0.23])
ax_explorer.set_title("DICOM Explorer", color='#00bcd4', fontsize=8, fontweight='bold')
ax_explorer.set_facecolor('#181818')
ax_explorer.set_xticks([]); ax_explorer.set_yticks([])

ax_exp_scroll = fig.add_axes([0.17, 0.13, 0.012, 0.23])
explorer_slider = Slider(ax=ax_exp_scroll, label='', valmin=0, valmax=0, valinit=0, valstep=1, orientation='vertical')
explorer_slider.ax.set_facecolor('#222222')
explorer_slider.on_changed(lambda val: (global_state.update({'explorer_scroll_offset': int(val)}), refresh_explorer_sidebar()))

# --- اللوحات الأربع الرئيسية (Grid 2x2) ---
ax1 = fig.add_axes([0.22, 0.54, 0.34, 0.37]) 
ax2 = fig.add_axes([0.62, 0.54, 0.34, 0.37]) 
ax3 = fig.add_axes([0.22, 0.13, 0.34, 0.37]) 
ax4 = fig.add_axes([0.62, 0.13, 0.34, 0.37]) 

panel_axes = {'p1': ax1, 'p2': ax2, 'p3': ax3, 'p4': ax4}
for p_key, ax in panel_axes.items(): ax.set_facecolor('black')

# أشرطة التمرير العمودية لكل شاشة عرض الشرائح
ax_scroll1 = fig.add_axes([0.57, 0.54, 0.012, 0.37])
scroll_p1 = Slider(ax=ax_scroll1, label='', valmin=0, valmax=nz-1, valinit=nz//2, valstep=1, orientation='vertical')
scroll_p1.ax.set_facecolor('#222222')

ax_scroll2 = fig.add_axes([0.97, 0.54, 0.012, 0.37])
scroll_p2 = Slider(ax=ax_scroll2, label='', valmin=0, valmax=ny-1, valinit=ny//2, valstep=1, orientation='vertical')
scroll_p2.ax.set_facecolor('#222222')

ax_scroll3 = fig.add_axes([0.57, 0.13, 0.012, 0.37])
scroll_p3 = Slider(ax=ax_scroll3, label='', valmin=0, valmax=nx-1, valinit=nx//2, valstep=1, orientation='vertical')
scroll_p3.ax.set_facecolor('#222222')

ax_scroll4 = fig.add_axes([0.97, 0.13, 0.012, 0.37])
scroll_p4 = Slider(ax=ax_scroll4, label='', valmin=0, valmax=nz-1, valinit=nz//2, valstep=1, orientation='vertical')
scroll_p4.ax.set_facecolor('#222222')

scroll_sliders = {'p1': scroll_p1, 'p2': scroll_p2, 'p3': scroll_p3, 'p4': scroll_p4}
scroll_axes_dict = {'p1': ax_scroll1, 'p2': ax_scroll2, 'p3': ax_scroll3, 'p4': ax_scroll4}

scroll_p1.on_changed(lambda val: (not global_state['updating_slider'] and panels['p1'].update({'slice': int(val)}) or update_display()))
scroll_p2.on_changed(lambda val: (not global_state['updating_slider'] and panels['p2'].update({'slice': int(val)}) or update_display()))
scroll_p3.on_changed(lambda val: (not global_state['updating_slider'] and panels['p3'].update({'slice': int(val)}) or update_display()))
scroll_p4.on_changed(lambda val: (not global_state['updating_slider'] and panels['p4'].update({'slice': int(val)}) or update_display()))

# --- أزرار التدوير وتصدير التقرير ---
ax_b1 = fig.add_axes([0.18, 0.03, 0.07, 0.07])
btn_ax = Button(ax_b1, 'Rotate 90', color='#0d47a1', hovercolor='#1565c0')
for l in btn_ax.ax.texts: l.set_color('white'); l.set_fontsize(8); l.set_fontweight('bold')
btn_ax.on_clicked(lambda e: (panels[global_state['target_panel']].update({'rot': (panels[global_state['target_panel']]['rot'] + 90) % 360}), update_display()))

ax_button = fig.add_axes([0.85, 0.03, 0.13, 0.07])
export_button = Button(ax_button, 'Export Report', color='#ff8f00', hovercolor='#ffa000')
for l in export_button.ax.texts: l.set_color('black'); l.set_fontweight('bold')
export_button.on_clicked(lambda e: fig.savefig("workstation_report.png", dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor()))

# ==========================================
# 4. Full Quality MRI Display Function
# ==========================================
def update_display():
    if global_state.get('updating_slider', False):
        return
    global_state['updating_slider'] = True
    
    sz = global_state['box_size']
    bri = global_state['brightness']
    con = global_state['contrast']
    sha = global_state['sharpness']
    max_panel = global_state['maximized_panel']
    layout = global_state['layout_mode']
    
    try:
        for p_key, ax in panel_axes.items():
            sc_ax = scroll_axes_dict[p_key]
            sc_ax.set_visible(False)
            
            if max_panel is not None:
                if p_key == max_panel:
                    ax.set_position([0.22, 0.13, 0.72, 0.78])
                    ax.set_visible(True)
                else:
                    ax.set_position([0, 0, 0.001, 0.001])
                    ax.set_visible(False)
                    continue
            else:
                if layout == '2x2':
                    if p_key == 'p1': ax.set_position([0.22, 0.54, 0.34, 0.37])
                    elif p_key == 'p2': ax.set_position([0.62, 0.54, 0.34, 0.37])
                    elif p_key == 'p3': ax.set_position([0.22, 0.13, 0.34, 0.37])
                    elif p_key == 'p4': ax.set_position([0.62, 0.13, 0.34, 0.37])
                    ax.set_visible(True)
                elif layout == '1x2':
                    if p_key == 'p1': ax.set_position([0.22, 0.13, 0.35, 0.78])
                    elif p_key == 'p2': ax.set_position([0.59, 0.13, 0.35, 0.78])
                    else:
                        ax.set_position([0, 0, 0.001, 0.001])
                        ax.set_visible(False)
                        continue
                    ax.set_visible(True)
                elif layout == '1x1':
                    if p_key == global_state['target_panel']:
                        ax.set_position([0.22, 0.13, 0.72, 0.78])
                        ax.set_visible(True)
                    else:
                        ax.set_position([0, 0, 0.001, 0.001])
                        ax.set_visible(False)
                        continue

            p_info = panels[p_key]
            seq_name = p_info['seq']
            data = loaded_sequences[seq_name]
            
            if data.ndim == 2:
                data = np.expand_dims(data, axis=-1)
            
            d_nx, d_ny, d_nz = data.shape
            max_slc = d_nz - 1
            p_info['slice'] = min(max(0, p_info['slice']), max_slc)
            
            ax.clear()
            is_target = (global_state['target_panel'] == p_key)
            border_color = '#ffcc00' if is_target else '#333333'
            border_width = 3.5 if is_target else 1.5
            for spine in ax.spines.values():
                spine.set_edgecolor(border_color)
                spine.set_linewidth(border_width)
                spine.set_visible(True)
                
            ax.set_xticks([]); ax.set_yticks([])
            
            im_slice = data[:, :, p_info['slice']].T
            
            processed = im_slice * con + bri
            if sha != 1.0:
                blurred = gaussian_filter(processed, sigma=1.0)
                processed = processed + (processed - blurred) * (sha - 1.0)
            im_slice = np.clip(processed, np.min(im_slice), np.max(im_slice))
            
            if p_info['rot'] != 0:
                im_slice = rotate(im_slice, p_info['rot'], reshape=True, mode='constant', cval=0)

            ax.imshow(im_slice, cmap='gray', origin='lower')
            title_str = f"[{p_key.upper()}] {seq_name} | Slice: {p_info['slice']}/{max_slc}"
            
            if global_state['box_active']:
                ax.add_patch(Rectangle((p_info['cx'] - sz//2, p_info['cy'] - sz//2), sz, sz, linewidth=2, edgecolor='cyan', facecolor='none'))

            if max_panel is None or max_panel == p_key:
                sc_ax.set_visible(True)
                if max_panel == p_key or layout == '1x1':
                    sc_ax.set_position([0.95, 0.13, 0.012, 0.78])
                elif layout == '1x2':
                    sc_ax.set_position([0.58 if p_key == 'p1' else 0.95, 0.13, 0.012, 0.78])
                else:
                    pos_map = {'p1': [0.57, 0.54], 'p2': [0.97, 0.54], 'p3': [0.57, 0.13], 'p4': [0.97, 0.13]}
                    sc_ax.set_position([pos_map[p_key][0], pos_map[p_key][1], 0.012, 0.37])
                scroll_sliders[p_key].valmax = max(1, max_slc)
                scroll_sliders[p_key].set_val(p_info['slice'])

            ax.set_title(title_str + (" [ACTIVE]" if is_target else ""), color='#ffcc00' if is_target else '#aaaaaa', fontsize=8, fontweight='bold' if is_target else 'normal')

        refresh_explorer_sidebar()
        t_info = panels[global_state['target_panel']]
        if global_state['box_active']:
            t_data = loaded_sequences[t_info['seq']]
            if t_data.ndim == 2: t_data = np.expand_dims(t_data, axis=-1)
            d_nx, d_ny, d_nz = t_data.shape
            half = global_state['box_size'] // 2
            region = t_data[max(0, t_info['cx'] - half):min(d_nx, t_info['cx'] + half), max(0, t_info['cy'] - half):min(d_ny, t_info['cy'] + half), min(max(0, t_info['cz']), d_nz - 1)]
            mean_val, std_val = (np.mean(region), np.std(region)) if region.size > 0 else (0.0, 0.0)
            meta_text.set_text(f"MRI STATS\n--------------------\nTarget : {global_state['target_panel'].upper()}\nCenter : ({t_info['cx']}, {t_info['cy']}, {t_info['cz']})\nMean   : {mean_val:.1f}\nStd    : {std_val:.1f}\nSNR    : {(mean_val/std_val if std_val > 0 else 0):.2f}")
        else:
            meta_text.set_text("METADATA\n--------------------\nBox is OFF")

        fig.canvas.draw_idle()
    finally:
        global_state['updating_slider'] = False

# ==========================================
# 5. DICOM Explorer Sidebar with Thumbnails
# ==========================================
thumb_axes_cache = []

def refresh_explorer_sidebar():
    global thumb_axes_cache
    for ax_item in thumb_axes_cache:
        try: ax_item.remove()
        except: pass
    thumb_axes_cache = []
    
    ax_explorer.clear()
    ax_explorer.set_title("DICOM Explorer", color='#00bcd4', fontsize=8, fontweight='bold')
    ax_explorer.set_facecolor('#181818')
    ax_explorer.set_xticks([]); ax_explorer.set_yticks([])
    
    n_seqs = len(seq_names)
    max_visible = 3
    offset = min(global_state['explorer_scroll_offset'], max(0, n_seqs - max_visible))
    
    if n_seqs > max_visible:
        explorer_slider.ax.set_visible(True)
        explorer_slider.valmax = n_seqs - max_visible
    else:
        explorer_slider.ax.set_visible(False)

    active_seq = panels[global_state['target_panel']]['seq']

    for i in range(offset, min(n_seqs, offset + max_visible)):
        s_name = seq_names[i]
        rel_idx = i - offset
        
        thumb_ax = fig.add_axes([0.03, 0.31 - (rel_idx * 0.075), 0.05, 0.06])
        thumb_axes_cache.append(thumb_ax)
        
        vol_data = loaded_sequences[s_name]
        th_slice = vol_data if vol_data.ndim == 2 else vol_data[:, :, vol_data.shape[2] // 2]
        
        thumb_ax.imshow(th_slice.T, cmap='gray', origin='lower')
        thumb_ax.set_xticks([]); thumb_ax.set_yticks([])
        
        is_selected = (s_name == active_seq)
        border_col = '#ff1744' if is_selected else '#00bcd4'
        border_lw = 2.5 if is_selected else 1.0
        for spine in thumb_ax.spines.values():
            spine.set_edgecolor(border_col)
            spine.set_linewidth(border_lw)
            
        thumb_ax.s_name_tag = s_name
        ax_explorer.text(0.38, 0.82 - (rel_idx * 0.31), f"#{i+1}: {s_name}", color='#ff1744' if is_selected else 'white', fontsize=7, transform=ax_explorer.transAxes, fontweight='bold' if is_selected else 'normal')

# ==========================================
# 6. Mouse & Keyboard Events
# ==========================================
def on_press(event):
    if event.inaxes in thumb_axes_cache and hasattr(event.inaxes, 's_name_tag'):
        panels[global_state['target_panel']]['seq'] = event.inaxes.s_name_tag
        update_display()
        return

    if event.dblclick and event.inaxes in panel_axes.values():
        for p_key, ax in panel_axes.items():
            if event.inaxes == ax:
                global_state['target_panel'] = p_key
                if global_state['maximized_panel'] == p_key:
                    global_state['maximized_panel'] = None
                    global_state['layout_mode'] = '2x2'
                else:
                    global_state['maximized_panel'] = p_key
                    global_state['layout_mode'] = '1x1'
                update_display()
                return

    if event.inaxes in panel_axes.values() and event.xdata is not None and event.ydata is not None:
        for p_key, ax in panel_axes.items():
            if event.inaxes == ax:
                global_state['target_panel'] = p_key
                t_info = panels[p_key]
                t_info['cx'], t_info['cy'] = int(event.xdata), int(event.ydata)
                update_display()
                break

fig.canvas.mpl_connect('button_press_event', on_press)

def on_scroll(event):
    if event.inaxes == ax_explorer or event.inaxes == explorer_slider.ax:
        curr = global_state['explorer_scroll_offset']
        global_state['explorer_scroll_offset'] = max(0, curr - 1) if event.button == 'up' else curr + 1
        explorer_slider.set_val(global_state['explorer_scroll_offset'])
        refresh_explorer_sidebar()
        fig.canvas.draw_idle()
    elif event.inaxes in panel_axes.values():
        for p_key, ax in panel_axes.items():
            if event.inaxes == ax:
                global_state['target_panel'] = p_key
                panels[p_key]['slice'] += (1 if event.button == 'up' else -1)
                update_display()
                break

fig.canvas.mpl_connect('scroll_event', on_scroll)
fig.canvas.mpl_connect('key_press_event', lambda event: panels[global_state['target_panel']].update({'slice': panels[global_state['target_panel']]['slice'] + (1 if event.key == 'up' else -1)}) or update_display() if event.key in ['up', 'down'] else None)

fig.canvas.draw()
update_display()
plt.show()
