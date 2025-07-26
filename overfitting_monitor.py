import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import display, clear_output
import time
import warnings
plt.rcParams['font.sans-serif'] = ['SimHei']  # 支持中文

class TrainingMonitor:
    def __init__(self, metrics=['loss'], figsize=(10, 6), window_size=5, risk_threshold=0.1):
        """
        训练过程监控器
        
        参数:
            metrics: 要监控的指标列表，如['loss', 'accuracy']
            figsize: 图表大小
            window_size: 计算波动率的窗口大小
            risk_threshold: 过拟合风险阈值(0-1)
        """
        self.metrics = metrics
        self.figsize = figsize
        self.window_size = window_size
        self.risk_threshold = risk_threshold
        
        # 初始化数据存储
        self.epochs = []
        self.train_data = {m: [] for m in metrics}
        self.test_data = {m: [] for m in metrics}
        
        # 创建图表
        self.fig, self.axs = plt.subplots(len(metrics), 1, figsize=figsize, squeeze=False)
        self.lines = {}
        
        for i, metric in enumerate(metrics):
            ax = self.axs[i, 0]
            ax.set_xlabel('Epoch')
            ax.set_ylabel(metric.capitalize())
            ax.set_title(f'{metric.capitalize()} Curve')
            ax.grid(True, alpha=0.3)
            
            # 初始化线条
            train_line, = ax.plot([], [], 'b-', linewidth=2, label='Train')
            test_line, = ax.plot([], [], 'r-', linewidth=2, label='Test')
            ax.legend()
            
            # 添加警告文本
            warning_text = ax.text(0.7, 0.9, '', transform=ax.transAxes, 
                                 bbox=dict(facecolor='white', alpha=0.7))
            
            self.lines[metric] = {
                'train': train_line,
                'test': test_line,
                'warning': warning_text
            }
        
        plt.tight_layout()
        self.is_ipython = 'inline' in plt.get_backend()
        self.last_update = time.time()
    
    def update(self, epoch, train_metrics, test_metrics):
        """更新监控数据"""
        self.epochs.append(epoch)
        
        for metric in self.metrics:
            self.train_data[metric].append(train_metrics[metric])
            self.test_data[metric].append(test_metrics[metric])
        
        # 实时更新图表（每0.5秒或新epoch时更新）
        if time.time() - self.last_update > 0.5 or epoch == self.epochs[-1]:
            self._update_plots()
            self._check_overfitting()
            self.last_update = time.time()
    
    def _update_plots(self):
        """更新所有指标的图表"""
        for metric in self.metrics:
            lines = self.lines[metric]
            lines['train'].set_data(self.epochs, self.train_data[metric])
            lines['test'].set_data(self.epochs, self.test_data[metric])
            
            # 调整坐标轴范围
            ax = lines['train'].axes
            ax.relim()
            ax.autoscale_view()
        
        if self.is_ipython:
            clear_output(wait=True)
            display(self.fig)
        else:
            plt.pause(0.01)
    
    def _check_overfitting(self):
        """检查过拟合风险"""
        for metric in self.metrics:
            if len(self.epochs) < self.window_size * 2:
                continue
                
            # 计算最近窗口期的平均差距
            recent_train = np.array(self.train_data[metric][-self.window_size:])
            recent_test = np.array(self.test_data[metric][-self.window_size:])
            gap = np.mean(recent_train - recent_test)
            
            # 计算测试集指标的波动率
            test_var = np.std(recent_test)
            
            # 判断过拟合风险
            risk_level = 0
            warning_msg = ''
            
            if gap > self.risk_threshold:
                risk_level = min(1, gap / (2 * self.risk_threshold))
                warning_msg = f'过拟合风险: {risk_level*100:.1f}%'
                
                if test_var > 0.05:  # 高波动率
                    warning_msg += ' (高波动)'
                elif gap > 2 * self.risk_threshold:
                    warning_msg += ' (严重)'
            
            # 更新警告显示
            if risk_level > 0.5:
                self.lines[metric]['warning'].set_color('red')
            elif risk_level > 0.3:
                self.lines[metric]['warning'].set_color('orange')
            else:
                self.lines[metric]['warning'].set_color('black')
                
            self.lines[metric]['warning'].set_text(warning_msg)
            
            # 高风险时发出警告
            if risk_level > 0.7:
                warnings.warn(f"高过拟合风险检测到! 指标: {metric}, 风险等级: {risk_level*100:.1f}%")
    
    def final_report(self):
        """生成最终报告"""
        print("\n=== 训练分析报告 ===")
        for metric in self.metrics:
            best_epoch = np.argmin(self.test_data[metric])
            print(f"\n指标 [{metric}]:")
            print(f"  最佳测试值: {self.test_data[metric][best_epoch]:.4f} (epoch {best_epoch+1})")
            print(f"  最终训练值: {self.train_data[metric][-1]:.4f}")
            print(f"  最终测试值: {self.test_data[metric][-1]:.4f}")
            
            final_gap = self.train_data[metric][-1] - self.test_data[metric][-1]
            if final_gap > self.risk_threshold:
                print(f"  ⚠️ 最终泛化间隙: {final_gap:.4f} (可能存在过拟合)")
        
        plt.show()


################# 以下是使用示例：###############
# 初始化监控器
monitor = TrainingMonitor(metrics=['loss', 'accuracy'], window_size=5)

# 模拟训练过程
for epoch in range(50):
    # 模拟训练和测试指标（实际使用时替换为真实数据）
    train_loss = np.exp(-epoch/10) + np.random.normal(0, 0.02)
    test_loss = 0.2 + 0.5*np.exp(-epoch/8) + np.random.normal(0, 0.03)
    
    train_acc = 1 - 0.5*np.exp(-epoch/15) + np.random.normal(0, 0.01)
    test_acc = 0.9 - 0.3*np.exp(-epoch/20) + np.random.normal(0, 0.02)
    
    # 更新监控器
    monitor.update(epoch, 
                  {'loss': train_loss, 'accuracy': train_acc},
                  {'loss': test_loss, 'accuracy': test_acc})
    
    # 模拟训练延迟
    time.sleep(0.1)

# 生成最终报告
monitor.final_report()