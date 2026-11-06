import os
import tkinter as tk
from tkinter import filedialog
import numpy as np
import nibabel as nib
import pydicom
import matplotlib.pyplot as plt
from matplotlib.widgets import CheckButtons, Slider, Button, RadioButtons
from matplotlib.patches import Rectangle
from scipy.ndimage import gaussian_filter

# إخفاء نافذة تكنسر الرئيسية
root = tk.Tk()
root.withdraw()

# ==========================================
# 1. Universal Medical Image Loader
# ==========================================
def load_medical_image(path):
    if os.path.isdir(path):
        dicom_files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.dcm') or '.' not in f]
        datasets = []
        for f in dicom_files:
            try:
                datasets.append(pydicom.dcmread(f))
            except:
                pass
        if not datasets:
            raise ValueError("No valid DICOM files found in the directory.")
        try:
            datasets.sort(key=lambda x: int(getattr(x, 'InstanceNumber', 0)))
        except:
            pass
        volume = np.stack([ds.pixel_array for ds in datasets], axis=-1)
        seq_name = getattr(datasets[0], 'SeriesDescription', 'DICOM_SERIES').upper()
        return volume, seq_name
    elif path.endswith(('.nii', '.nii.gz')):
        img = nib.load(path)
        volume = img.get_fdata()
        seq_name = os.path.splitext(os.path.basename(path))[0].upper().replace('.NII', '')
        return volume, seq_name
    elif path.endswith('.dcm') or '.' not in path:
        try:
            ds = pydicom.dcmread(path)
            volume = ds.pixel_array
            if volume.ndim == 2:
                volume = np.expand_dims(volume, axis=-1)
            seq_name = getattr(ds, 'SeriesDescription', 'DICOM_SINGLE').upper()
            return volume, seq_name
        except Exception as e:
            raise ValueError(f"Could not read DICOM file: {e}")
    else:
        raise ValueError("Unsupported file format.")

# تحميل السيكونس الأساسي مع الفحص الذكي
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
        title="Select Medical Image(s) to Start Workstation",
        filetypes=[
            ("Medical Images", "*.nii;*.nii.gz;*.dcm"),
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
    loaded_sequences['DEFAULT_BRAIN'] = np.random.rand(240, 240, 155)

seq_names = list(loaded_sequences.keys())
default_seq = seq_names[0]

vol_init = loaded_sequences[default_seq]
if vol_init.ndim == 2:
    vol_init = np.expand_dims(vol_init, axis=-1)
nx, ny, nz = vol_init.shape

# ==========================================
# 2. Flexible Multi-Panel & 3D Voxel State
# ==========================================
panels = {
    'p1': {'seq': default_seq, 'type': 'axial', 'slice': nz // 2},
    'p2': {'seq': default_seq, 'type': 'coronal', 'slice': ny // 2}, 
    'p3': {'seq': default_seq, 'type': 'sagittal', 'slice': nx // 2},
    'p4': {'seq': default_seq, 'type': 'mrs', 'slice': 0}          
}

global_state = {
    'target_panel': 'p1',       
    'box_active': True,
    'cx': nx // 2,              
    'cy': ny // 2,              
    'cz': nz // 2,              
    'box_size': 25,
    'brightness': 0.0,          # إعداد السطوع
    'contrast': 1.0,            # إعداد التباين
    'sharpness': 1.0,           # إعداد الحدة
    'updating_slider': False    
}

# ==========================================
# 3. Setup Professional Figure Layout & UI
# ==========================================
fig = plt.figure(figsize=(16, 9), facecolor='#121212')
fig.canvas.manager.set_window_title('Professional 3D Medical Workstation - PACS Interface')

# --- 1. زر إظهار/إخفاء مربع الـ Voxel Box ---
ax_check_box = fig.add_axes([0.02, 0.93, 0.16, 0.04])
check_box = CheckButtons(ax_check_box, ['Voxel Box'], [global_state['box_active']])
check_box.ax.set_facecolor('#1e1e1e')
for l in check_box.ax.texts:
    l.set_color('#00e5ff'); l.set_fontsize(8); l.set_fontweight('bold')

def toggle_box(label):
    global_state['box_active'] = not global_state['box_active']
    update_display()
check_box.on_clicked(toggle_box)

# --- 2. شريط تحكم حجم البوكس (Voxel Size) ---
ax_slider = fig.add_axes([0.02, 0.88, 0.16, 0.025])
size_slider = Slider(ax=ax_slider, label='Voxel Size', valmin=5, valmax=80, valinit=global_state['box_size'], valstep=1)
for l in size_slider.ax.texts: 
    l.set_color('#00e5ff'); l.set_fontsize(7); l.set_fontweight('bold')
size_slider.ax.set_facecolor('#1e1e1e')

def update_size(val):
    global_state['box_size'] = int(val)
    update_display()
size_slider.on_changed(update_size)

# --- 3. أشرطة التحكم الجديدة: Brightness, Contrast, Sharpness ---
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

# --- زر إعادة الضبط (Reset Button) ---
ax_reset = fig.add_axes([0.02, 0.70, 0.16, 0.035])
btn_reset = Button(ax_reset, 'Reset Adjustments', color='#b71c1c', hovercolor='#c62828')
for l in btn_reset.ax.texts: l.set_color('white'); l.set_fontsize(7); l.set_fontweight('bold')

def reset_adjustments(event):
    global_state['brightness'] = 0.0
    global_state['contrast'] = 1.0
    global_state['sharpness'] = 1.0
    slider_bri.set_val(0.0)
    slider_con.set_val(1.0)
    slider_sha.set_val(1.0)
    update_display()
btn_reset.on_clicked(reset_adjustments)

# --- 4. شريط البيانات والثابت تحته مباشرة ---
ax_meta = fig.add_axes([0.02, 0.45, 0.16, 0.23])
ax_meta.set_facecolor('#181818')
for spine in ax_meta.spines.values():
    spine.set_edgecolor('#00bcd4')
    spine.set_linewidth(1.5)
ax_meta.set_xticks([])
ax_meta.set_yticks([])
meta_text = ax_meta.text(0.05, 0.5, "", color='#00e5ff', fontsize=9, fontweight='bold', va='center', family='monospace')

# --- 5. زر فتح ملفات متعددة ---
ax_open_btn = fig.add_axes([0.02, 0.38, 0.16, 0.05])
open_button = Button(ax_open_btn, '+ Open & Add Files', color='#1b5e20', hovercolor='#2e7d32')
for lbl in open_button.ax.texts:
    lbl.set_color('white'); lbl.set_fontsize(8); lbl.set_fontweight('bold')

def open_and_add_files(event):
    global seq_names
    file_paths = filedialog.askopenfilenames(
        title="Select Medical Images (DICOM & NIfTI)",
        filetypes=[
            ("Medical Images", "*.nii;*.nii.gz;*.dcm"),
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
                print(f"[ERROR] Could not load {file_path}: {e}")
        
        if added_count > 0:
            refresh_sidebar()
            update_display()

open_button.on_clicked(open_and_add_files)

# --- 6. الشريط الجانبي لاختيار الصور في الأسفل ---
ax_sidebar = fig.add_axes([0.02, 0.13, 0.16, 0.23])
ax_sidebar.set_title("Loaded Sequences", color='#00bcd4', fontsize=8, fontweight='bold')
ax_sidebar.set_facecolor('#1e1e1e')

# --- توحيد أحجام ومقاسات الشاشات الأربع ---
ax1 = fig.add_axes([0.22, 0.54, 0.34, 0.37]) 
ax2 = fig.add_axes([0.62, 0.54, 0.34, 0.37]) 
ax3 = fig.add_axes([0.22, 0.13, 0.34, 0.37]) 
ax4 = fig.add_axes([0.62, 0.13, 0.34, 0.37]) 

panel_axes = {'p1': ax1, 'p2': ax2, 'p3': ax3, 'p4': ax4}

for p_key, ax in panel_axes.items():
    ax.set_facecolor('black')

# أشرطة التمرير العمودية (Vertical Scrollbars) بجانب كل شاشة
ax_scroll1 = fig.add_axes([0.57, 0.54, 0.012, 0.37])
scroll_p1 = Slider(ax=ax_scroll1, label='', valmin=0, valmax=nz-1, valinit=nz//2, valstep=1, orientation='vertical')
scroll_p1.ax.set_facecolor('#222222')

ax_scroll2 = fig.add_axes([0.97, 0.54, 0.012, 0.37])
scroll_p2 = Slider(ax=ax_scroll2, label='', valmin=0, valmax=ny-1, valinit=ny//2, valstep=1, orientation='vertical')
scroll_p2.ax.set_facecolor('#222222')

ax_scroll3 = fig.add_axes([0.57, 0.13, 0.012, 0.37])
scroll_p3 = Slider(ax=ax_scroll3, label='', valmin=0, valmax=nx-1, valinit=nx//2, valstep=1, orientation='vertical')
scroll_p3.ax.set_facecolor('#222222')

scroll_sliders = {'p1': scroll_p1, 'p2': scroll_p2, 'p3': scroll_p3}

def on_scroll_slider_p1(val):
    if not global_state['updating_slider'] and panels['p1']['type'] != 'mrs':
        global_state['cz'] = int(val)
        panels['p1']['slice'] = int(val)
        update_display()

def on_scroll_slider_p2(val):
    if not global_state['updating_slider'] and panels['p2']['type'] != 'mrs':
        global_state['cy'] = int(val)
        panels['p2']['slice'] = int(val)
        update_display()

def on_scroll_slider_p3(val):
    if not global_state['updating_slider'] and panels['p3']['type'] != 'mrs':
        global_state['cx'] = int(val)
        panels['p3']['slice'] = int(val)
        update_display()

scroll_p1.on_changed(on_scroll_slider_p1)
scroll_p2.on_changed(on_scroll_slider_p2)
scroll_p3.on_changed(on_scroll_slider_p3)

# ==========================================
# 4. Bottom Control Panel Definitions & Highlighting
# ==========================================
ax_b1 = fig.add_axes([0.18, 0.03, 0.09, 0.07])
btn_ax = Button(ax_b1, 'As Axial', color='#0d47a1', hovercolor='#1565c0')
for l in btn_ax.ax.texts: l.set_color('white'); l.set_fontsize(8); l.set_fontweight('bold')

ax_b2 = fig.add_axes([0.28, 0.03, 0.09, 0.07])
btn_cor = Button(ax_b2, 'As Coronal', color='#0d47a1', hovercolor='#1565c0')
for l in btn_cor.ax.texts: l.set_color('white'); l.set_fontsize(8); l.set_fontweight('bold')

ax_b3 = fig.add_axes([0.38, 0.03, 0.09, 0.07])
btn_sag = Button(ax_b3, 'As Sagittal', color='#0d47a1', hovercolor='#1565c0')
for l in btn_sag.ax.texts: l.set_color('white'); l.set_fontsize(8); l.set_fontweight('bold')

ax_b4 = fig.add_axes([0.48, 0.03, 0.09, 0.07])
btn_mrs = Button(ax_b4, 'As MRS', color='#4a148c', hovercolor='#8e24aa')
for l in btn_mrs.ax.texts: l.set_color('white'); l.set_fontsize(8); l.set_fontweight('bold')

def update_buttons_highlight():
    t_type = panels[global_state['target_panel']]['type']
    buttons_map = {
        'axial': (btn_ax, '#0d47a1'),
        'coronal': (btn_cor, '#0d47a1'),
        'sagittal': (btn_sag, '#0d47a1'),
        'mrs': (btn_mrs, '#4a148c')
    }
    
    for b_type, (btn, default_bg) in buttons_map.items():
        if btn is not None:
            if b_type == t_type:
                btn.ax.set_facecolor('#ffcc00')  # تظليل ذهبي دقيق ومتزامن بالأسفل
                btn.color = '#ffcc00'
                for l in btn.ax.texts: 
                    l.set_color('black')
                    l.set_fontweight('bold')
            else:
                btn.ax.set_facecolor(default_bg)
                btn.color = default_bg
                for l in btn.ax.texts: 
                    l.set_color('white')
                    l.set_fontweight('normal')

def set_type_axial(event): panels[global_state['target_panel']]['type'] = 'axial'; update_display()
def set_type_coronal(event): panels[global_state['target_panel']]['type'] = 'coronal'; update_display()
def set_type_sagittal(event): panels[global_state['target_panel']]['type'] = 'sagittal'; update_display()
def set_type_mrs(event): panels[global_state['target_panel']]['type'] = 'mrs'; update_display()

btn_ax.on_clicked(set_type_axial)
btn_cor.on_clicked(set_type_coronal)
btn_sag.on_clicked(set_type_sagittal)
btn_mrs.on_clicked(set_type_mrs)

ax_button = fig.add_axes([0.85, 0.03, 0.13, 0.07])
export_button = Button(ax_button, 'Export Report', color='#ff8f00', hovercolor='#ffa000')
for l in export_button.ax.texts: l.set_color('black'); l.set_fontweight('bold')

def export_report(event):
    report_img_path = "workstation_report.png"
    fig.savefig(report_img_path, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
    print(f"[SUCCESS] Report saved to: {os.path.abspath(report_img_path)}")

export_button.on_clicked(export_report)

# ==========================================
# 5. Update Display Function with Enhancements
# ==========================================
def update_display():
    global_state['updating_slider'] = True
    cx, cy, cz = global_state['cx'], global_state['cy'], global_state['cz']
    sz = global_state['box_size']
    bri = global_state['brightness']
    con = global_state['contrast']
    sha = global_state['sharpness']
    
    try:
        for p_key, ax in panel_axes.items():
            p_info = panels[p_key]
            seq_name = p_info['seq']
            data = loaded_sequences[seq_name]
            
            if data.ndim == 2:
                data = np.expand_dims(data, axis=-1)
            
            d_nx, d_ny, d_nz = data.shape
            
            ax.clear()
            is_target = (global_state['target_panel'] == p_key)
            
            border_color = '#ffcc00' if is_target else '#333333'
            border_width = 3.5 if is_target else 1.5
            for spine in ax.spines.values():
                spine.set_edgecolor(border_color)
                spine.set_linewidth(border_width)
                spine.set_visible(True)
                
            ax.set_xticks([])
            ax.set_yticks([])
            
            # Helper function for applying Brightness, Contrast & Sharpness
            def enhance_slice(slc):
                # Contrast & Brightness
                processed = slc * con + bri
                # Sharpness via unsharp mask / gaussian filter difference
                if sha != 1.0:
                    blurred = gaussian_filter(processed, sigma=1.0)
                    processed = processed + (processed - blurred) * (sha - 1.0)
                return np.clip(processed, np.min(slc), np.max(slc))

            # 1. Axial View
            if p_info['type'] == 'axial':
                p_info['slice'] = min(max(0, cz), d_nz - 1)
                im_slice = data[:, :, p_info['slice']].T
                im_slice = enhance_slice(im_slice)
                ax.imshow(im_slice, cmap='gray', origin='lower')
                title_str = f"[{p_key.upper()}] {seq_name} | Axial (Z): {p_info['slice']}/{d_nz-1}"
                
                if global_state['box_active']:
                    rect = Rectangle((cx - sz//2, cy - sz//2), sz, sz, linewidth=2, edgecolor='cyan', facecolor='none')
                    ax.add_patch(rect)
                if p_key in scroll_sliders:
                    scroll_sliders[p_key].ax.set_visible(True)
                    scroll_sliders[p_key].valmax = max(1, d_nz - 1)
                    scroll_sliders[p_key].set_val(p_info['slice'])
                    
            # 2. Coronal View
            elif p_info['type'] == 'coronal':
                p_info['slice'] = min(max(0, cy), d_ny - 1)
                im_slice = data[:, p_info['slice'], :].T
                im_slice = enhance_slice(im_slice)
                ax.imshow(im_slice, cmap='gray', origin='lower')
                title_str = f"[{p_key.upper()}] {seq_name} | Coronal (Y): {p_info['slice']}/{d_ny-1}"
                
                if global_state['box_active']:
                    rect = Rectangle((cx - sz//2, cz - sz//2), sz, sz, linewidth=2, edgecolor='cyan', facecolor='none')
                    ax.add_patch(rect)
                if p_key in scroll_sliders:
                    scroll_sliders[p_key].ax.set_visible(True)
                    scroll_sliders[p_key].valmax = max(1, d_ny - 1)
                    scroll_sliders[p_key].set_val(p_info['slice'])
                
            # 3. Sagittal View
            elif p_info['type'] == 'sagittal':
                p_info['slice'] = min(max(0, cx), d_nx - 1)
                im_slice = data[p_info['slice'], :, :].T
                im_slice = enhance_slice(im_slice)
                ax.imshow(im_slice, cmap='gray', origin='lower')
                title_str = f"[{p_key.upper()}] {seq_name} | Sagittal (X): {p_info['slice']}/{d_nx-1}"
                
                if global_state['box_active']:
                    rect = Rectangle((cy - sz//2, cz - sz//2), sz, sz, linewidth=2, edgecolor='cyan', facecolor='none')
                    ax.add_patch(rect)
                if p_key in scroll_sliders:
                    scroll_sliders[p_key].ax.set_visible(True)
                    scroll_sliders[p_key].valmax = max(1, d_nx - 1)
                    scroll_sliders[p_key].set_val(p_info['slice'])
                
            # 4. MRS Spectrum View
            elif p_info['type'] == 'mrs':
                if p_key in scroll_sliders:
                    scroll_sliders[p_key].ax.set_visible(False)
                    
                x_freq = np.linspace(0.5, 4.0, 300)
                half = sz // 2
                region = data[max(0, cx-half):min(d_nx, cx+half), max(0, cy-half):min(d_ny, cy+half), min(d_nz-1, cz)]
                m_val = np.mean(region) if region.size > 0 else 50.0
                t_fact = m_val / 100.0
                
                naa = 2.2 * t_fact * np.exp(-((x_freq - 2.0)**2) / 0.015)
                cr  = 1.5 * t_fact * np.exp(-((x_freq - 3.0)**2) / 0.015)
                cho = (1.8 + t_fact*0.5) * np.exp(-((x_freq - 3.2)**2) / 0.015)
                lipid = 0.8 * t_fact * np.exp(-((x_freq - 1.3)**2) / 0.025)
                spec_y = naa + cr + cho + lipid + np.random.normal(0, 0.02, size=x_freq.shape)
                
                ax.plot(x_freq, spec_y, color='lime', lw=2)
                ax.invert_xaxis()
                
                ax.annotate('NAA', xy=(2.0, np.max(naa)), xytext=(1.8, 3.2),
                             arrowprops=dict(facecolor='cyan', shrink=0.05, width=1, headwidth=4), color='cyan', fontsize=7, fontweight='bold')
                ax.annotate('Creatine', xy=(3.0, np.max(cr)), xytext=(2.6, 2.8),
                             arrowprops=dict(facecolor='orange', shrink=0.05, width=1, headwidth=4), color='orange', fontsize=7, fontweight='bold')
                ax.annotate('Choline', xy=(3.2, np.max(cho)), xytext=(3.3, 3.5),
                             arrowprops=dict(facecolor='magenta', shrink=0.05, width=1, headwidth=4), color='magenta', fontsize=7, fontweight='bold')
                
                ax.set_facecolor('black')
                ax.tick_params(colors='white')
                title_str = f"[{p_key.upper()}] MRS Spectrum [{seq_name}]"

            if is_target:
                ax.set_title(title_str + " [ACTIVE TARGET]", color='#ffcc00', fontsize=9, fontweight='bold')
            else:
                ax.set_title(title_str, color='#aaaaaa', fontsize=9)

        update_buttons_highlight()

        # --- تحديث شريط البيانات الثابت في الأعلى ---
        t_info = panels[global_state['target_panel']]
        if t_info['type'] != 'mrs' and global_state['box_active']:
            t_data = loaded_sequences[t_info['seq']]
            if t_data.ndim == 2:
                t_data = np.expand_dims(t_data, axis=-1)
            d_nx, d_ny, d_nz = t_data.shape
            half = global_state['box_size'] // 2
            xmin, xmax = max(0, cx - half), min(d_nx, cx + half)
            ymin, ymax = max(0, cy - half), min(d_ny, cy + half)
            z_idx = min(max(0, cz), d_nz - 1)
            
            region = t_data[xmin:xmax, ymin:ymax, z_idx]
            mean_val = np.mean(region) if region.size > 0 else 0.0
            std_val = np.std(region) if region.size > 0 else 0.0
            snr_val = (mean_val / std_val) if std_val > 0 else 0.0
            
            meta_str = (
                f"METADATA & STATS\n"
                f"--------------------\n"
                f"Target : {global_state['target_panel'].upper()} ({t_info['type']})\n"
                f"Center : ({cx}, {cy}, {cz})\n"
                f"Size   : {global_state['box_size']}x{global_state['box_size']} px\n"
                f"Mean   : {mean_val:.1f}\n"
                f"Std    : {std_val:.1f}\n"
                f"SNR    : {snr_val:.2f}"
            )
            meta_text.set_text(meta_str)
        else:
            meta_text.set_text("METADATA\n--------------------\nBox is OFF or MRS Active")

        fig.canvas.draw_idle()
    finally:
        global_state['updating_slider'] = False

# ==========================================
# 6. Sidebar & Events
# ==========================================
radio_seq = None
def refresh_sidebar():
    global radio_seq
    ax_sidebar.clear()
    ax_sidebar.set_title(f"Target: {global_state['target_panel'].upper()}", color='#ffcc00', fontsize=8, fontweight='bold')
    ax_sidebar.set_facecolor('#1e1e1e')
    
    current_target_seq = panels[global_state['target_panel']]['seq']
    active_idx = seq_names.index(current_target_seq) if current_target_seq in seq_names else 0
    
    radio_seq = RadioButtons(ax_sidebar, seq_names, active=active_idx)
    for label in radio_seq.labels:
        label.set_color('white'); label.set_fontsize(8)
    radio_seq.on_clicked(on_sequence_assign)
    fig.canvas.draw_idle()

def on_sequence_assign(label):
    t_panel = global_state['target_panel']
    panels[t_panel]['seq'] = label
    update_display()

refresh_sidebar()

def on_click_select_panel(event):
    if event.inaxes == ax1: global_state['target_panel'] = 'p1'
    elif event.inaxes == ax2: global_state['target_panel'] = 'p2'
    elif event.inaxes == ax3: global_state['target_panel'] = 'p3'
    elif event.inaxes == ax4: global_state['target_panel'] = 'p4'
    else: return
    
    t_info = panels[global_state['target_panel']]
    if event.xdata is not None and event.ydata is not None:
        x, y = int(event.xdata), int(event.ydata)
        if t_info['type'] == 'axial':
            global_state['cx'], global_state['cy'] = x, y
        elif t_info['type'] == 'coronal':
            global_state['cx'], global_state['cz'] = x, y
        elif t_info['type'] == 'sagittal':
            global_state['cy'], global_state['cz'] = x, y
            
    refresh_sidebar()
    update_display()

fig.canvas.mpl_connect('button_press_event', on_click_select_panel)

def on_scroll(event):
    if event.inaxes in panel_axes.values():
        for p_key, ax in panel_axes.items():
            if event.inaxes == ax:
                global_state['target_panel'] = p_key
                t_info = panels[p_key]
                if t_info['type'] != 'mrs':
                    if event.button == 'up':
                        if t_info['type'] == 'axial': global_state['cz'] += 1
                        elif t_info['type'] == 'coronal': global_state['cy'] += 1
                        elif t_info['type'] == 'sagittal': global_state['cx'] += 1
                    elif event.button == 'down':
                        if t_info['type'] == 'axial': global_state['cz'] -= 1
                        elif t_info['type'] == 'coronal': global_state['cy'] -= 1
                        elif t_info['type'] == 'sagittal': global_state['cx'] -= 1
                    refresh_sidebar()
                    update_display()
                break

fig.canvas.mpl_connect('scroll_event', on_scroll)

def on_key(event):
    t_info = panels[global_state['target_panel']]
    if event.key == 'up':
        if t_info['type'] == 'axial': global_state['cz'] += 1
        elif t_info['type'] == 'coronal': global_state['cy'] += 1
        elif t_info['type'] == 'sagittal': global_state['cx'] += 1
    elif event.key == 'down':
        if t_info['type'] == 'axial': global_state['cz'] -= 1
        elif t_info['type'] == 'coronal': global_state['cy'] -= 1
        elif t_info['type'] == 'sagittal': global_state['cx'] -= 1
    update_display()

fig.canvas.mpl_connect('key_press_event', on_key)

dragging = False
def on_press(event):
    global dragging
    t_panel = global_state['target_panel']
    t_info = panels[t_panel]
    if event.inaxes == panel_axes[t_panel] and t_info['type'] != 'mrs' and global_state['box_active']:
        dragging = True
        x, y = int(event.xdata), int(event.ydata)
        if t_info['type'] == 'axial':
            global_state['cx'], global_state['cy'] = x, y
        elif t_info['type'] == 'coronal':
            global_state['cx'], global_state['cz'] = x, y
        elif t_info['type'] == 'sagittal':
            global_state['cy'], global_state['cz'] = x, y
        update_display()

def on_motion(event):
    global dragging
    t_panel = global_state['target_panel']
    t_info = panels[t_panel]
    if dragging and event.inaxes == panel_axes[t_panel] and event.xdata is not None and event.ydata is not None:
        x, y = int(event.xdata), int(event.ydata)
        if t_info['type'] == 'axial':
            global_state['cx'], global_state['cy'] = x, y
        elif t_info['type'] == 'coronal':
            global_state['cx'], global_state['cz'] = x, y
        elif t_info['type'] == 'sagittal':
            global_state['cy'], global_state['cz'] = x, y
        update_display()

def on_release(event):
    global dragging
    dragging = False

fig.canvas.mpl_connect('button_press_event', on_press)
fig.canvas.mpl_connect('motion_notify_event', on_motion)
fig.canvas.mpl_connect('button_release_event', on_release)

update_display()
plt.show()