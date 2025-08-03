import numpy as np
from sklearn.metrics import confusion_matrix
from typing import Callable, Dict, Optional

class ThresholdOptimizer:
    """
    通用分类阈值优化器
    
    功能：
    1. 支持自定义优化目标函数（利润、F1等）
    2. 支持添加约束条件（如召回率下限）
    3. 内置混淆矩阵四种情况的快捷计算
    4. 支持多种阈值搜索策略
    
    示例：
    >>> optimizer = ThresholdOptimizer(
            objective=lambda tn, fp, fn, tp: tp*1000 - fp*200,
            constraints=[('recall', '>=', 0.8)]
        )
    >>> best_threshold = optimizer.find_optimal_threshold(model, X_val, y_val)
    """
    
    def __init__(self, 
                 objective: Callable[[int, int, int, int], float],
                 constraints: Optional[list] = None,
                 threshold_range: tuple = (0, 1),
                 step: float = 0.01):
        """
        初始化优化器
        
        参数：
        - objective: 目标函数，输入(TN, FP, FN, TP)，返回优化得分
        - constraints: 约束条件列表，每个元素为(指标名, 运算符, 值)
                      支持指标：'precision', 'recall', 'fpr', 'accuracy'
        - threshold_range: 阈值搜索范围
        - step: 搜索步长
        """
        self.objective = objective
        self.constraints = constraints or []
        self.threshold_range = threshold_range
        self.step = step
        
        # 内置指标计算字典
        self.metric_funcs = {
            'precision': self._calc_precision,
            'recall': self._calc_recall,
            'fpr': self._calc_fpr,
            'accuracy': self._calc_accuracy
        }
    
    def find_optimal_threshold(self, model, X, y_true):
        """寻找最优阈值"""
        # 获取预测概率（兼容不同sklearn版本）
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X)[:, 1]
        elif hasattr(model, "decision_function"):
            y_proba = model.decision_function(X)
        else:
            raise ValueError("模型必须提供 predict_proba 或 decision_function 方法")
        
        best_score = -np.inf
        best_threshold = 0.5  # 默认阈值
        best_y_pred = None
        
        # 遍历阈值
        for threshold in np.arange(*self.threshold_range, self.step):
            y_pred = (y_proba >= threshold).astype(int)
            
            # 计算混淆矩阵
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            
            # 检查约束条件
            if not self._check_constraints(tn, fp, fn, tp):
                continue
                
            # 计算目标得分
            current_score = self.objective(tn, fp, fn, tp)
            
            # 更新最佳结果
            if current_score > best_score:
                best_score = current_score
                best_threshold = threshold
                best_y_pred = y_pred
        
        return {
            'threshold': best_threshold,
            'score': best_score,
            'y_pred': best_y_pred,
            'confusion_matrix': (tn, fp, fn, tp) if best_y_pred is not None else None
        }
    
    def _check_constraints(self, tn: int, fp: int, fn: int, tp: int) -> bool:
        """检查是否满足所有约束条件"""
        for metric, operator, value in self.constraints:
            # 计算当前指标值
            metric_value = self.metric_funcs[metric](tn, fp, fn, tp)
            
            # 比较运算
            if operator == '>=' and not (metric_value >= value):
                return False
            elif operator == '<=' and not (metric_value <= value):
                return False
            elif operator == '>' and not (metric_value > value):
                return False
            elif operator == '<' and not (metric_value < value):
                return False
            elif operator == '==' and not (metric_value == value):
                return False
        return True
    
    # ========== 内置指标计算 ==========
    @staticmethod
    def _calc_precision(tn, fp, fn, tp):
        return tp / (tp + fp) if (tp + fp) > 0 else 0
    
    @staticmethod
    def _calc_recall(tn, fp, fn, tp):
        return tp / (tp + fn) if (tp + fn) > 0 else 0
    
    @staticmethod
    def _calc_fpr(tn, fp, fn, tp):
        return fp / (fp + tn) if (fp + tn) > 0 else 0
    
    @staticmethod
    def _calc_accuracy(tn, fp, fn, tp):
        return (tp + tn) / (tp + tn + fp + fn)
    
    # ========== 常用目标函数预设 ==========
    @classmethod
    def profit_objective(cls, tp_gain: float, fp_cost: float):
        """生成利润目标函数"""
        return lambda tn, fp, fn, tp: tp * tp_gain - fp * fp_cost
    
    @classmethod
    def fbeta_objective(cls, beta: float = 1):
        """生成F-beta目标函数"""
        return lambda tn, fp, fn, tp: (
            (1 + beta**2) * tp / ((1 + beta**2) * tp + fp + beta**2 * fn)
            if (tp + fp + fn) > 0 else 0
        )


# 使用示例
if __name__ == "__main__":
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    
    # 生成模拟数据
    X, y = make_classification(n_samples=1000, weights=[0.9, 0.1], random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # 训练模型
    model = RandomForestClassifier(random_state=42).fit(X_train, y_train)
    
    # 案例1：利润最大化（正确预测正类+1500，误判正类-500）
    optimizer1 = ThresholdOptimizer(
        objective=ThresholdOptimizer.profit_objective(tp_gain=1500, fp_cost=500),
        constraints=[('recall', '>=', 0.7)]  # 召回率不低于70%
    )
    result1 = optimizer1.find_optimal_threshold(model, X_val, y_val)
    print(f"利润最优阈值: {result1['threshold']:.2f}, 预期利润: {result1['score']:.0f}")
    
    # 案例2：F1分数最大化
    optimizer2 = ThresholdOptimizer(
        objective=ThresholdOptimizer.fbeta_objective(beta=1),  # F1-score
        threshold_range=(0.3, 0.8)  # 限定阈值搜索范围
    )
    result2 = optimizer2.find_optimal_threshold(model, X_val, y_val)
    print(f"F1最优阈值: {result2['threshold']:.2f}, F1-score: {result2['score']:.3f}")