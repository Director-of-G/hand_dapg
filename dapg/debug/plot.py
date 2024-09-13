import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pickle


finger_to_col_map = {"th": 4, "ff": 3, "mf": 2, "rf": 1, "lf": 0}
knuckle_to_row_map = {"distal": 0, "middle": 1, "proximal": 2, "fingertip": 0}
panel_to_idx_map = {}

def plot_tactile_readings(data, max_taxel_reading=1.0):
    fig, axes = plt.subplots(3, 5, figsize=(10, 6))
    for name in data.keys():
        finger = name.split('-')[0][:2]
        knuckle = name.split('-')[0][2:]
        row = knuckle_to_row_map[knuckle]
        col = finger_to_col_map[finger]
        ax = axes[row, col]
        im = ax.imshow(data[name], cmap='gray', vmin=0, vmax=max_taxel_reading)
        ax.axis('off')
        ax.set_title(name)
    
    plt.show()

if __name__ == "__main__":
    tactile_data = pickle.load(open('./real_tactile.pkl', 'rb'))

    n_timesteps = len(tactile_data)
    ani_interval = 25.0 * 1000 / n_timesteps
    image_size = (4, 4)  # 每幅图像的尺寸

    # 创建画布
    fig, axes = plt.subplots(3, 5, figsize=(10, 6))

    max_taxel_reading = 0
    for t in range(n_timesteps):
        for key in tactile_data[t].keys():
            if tactile_data[t][key].max() > max_taxel_reading:
                max_taxel_reading = tactile_data[t][key].max()

    # 初始化绘图
    im_plots = []
    im_txts = []
    for name in tactile_data[0].keys():
        cur_txt = []
        finger = name.split('-')[0][:2]
        knuckle = name.split('-')[0][2:]
        row = knuckle_to_row_map[knuckle]
        col = finger_to_col_map[finger]
        ax = axes[row, col]
        sensor_act_value = tactile_data[0][name]
        im = ax.imshow(sensor_act_value, cmap='gray', vmin=0, vmax=max_taxel_reading)
        
        for i in range(image_size[0]):
            row = []
            for j in range(image_size[1]):
                value = sensor_act_value[i][j]
                text_color = 'white' if value < 0.5 else 'black'
                text = ax.text(j, i, str(round(value, 1)), ha='center', va='center', color=text_color)
                row.append(text)
            cur_txt.append(row)
        im_txts.append(cur_txt)

        ax.axis('off')
        ax.set_title(name)
        panel_to_idx_map[name] = len(im_plots)
        im_plots.append(im)

    # 更新函数
    def update(t):
        print(f"timestep {t}")
        for name in tactile_data[t].keys():
            idx = panel_to_idx_map[name]
            sensor_act_value = tactile_data[t][name]
            im_plots[idx].set_array(sensor_act_value)
            for i in range(image_size[0]):
                for j in range(image_size[1]):
                    value = sensor_act_value[i, j]
                    # 根据像素值调整文本颜色
                    text_color = 'white' if value < 0.5 else 'black'
                    im_txts[idx][i][j].set_text(f'{value:.1f}')
                    im_txts[idx][i][j].set_color(text_color)
        return im_plots

    # 创建动画
    ani = animation.FuncAnimation(fig, update, frames=n_timesteps, interval=ani_interval, blit=True)

    # 保存动画
    ani.save('grayscale_animation.mp4', writer='ffmpeg')

    plt.show()
