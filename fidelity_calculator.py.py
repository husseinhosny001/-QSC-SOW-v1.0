fidelity_calculator.py النموذجي (محتوى أولي):

```python
"""
أداة حساب إخلاص التشابك من بيانات استعادة الحالة الكمومية (QST).
الإصدار: 0.1.0
"""
import numpy as np
from scipy.linalg import sqrtm

def calculate_density_matrix(counts_dict):
    """
    تحسب مصفوفة الكثافة 4x4 من قاموس العدود.
    
    المدخلات:
        counts_dict: قاموس بـ 16 مفتاحاً (مثل 'HH', 'HV', ... 'II')
        بقيم العدود المقابلة.
    
    المخرجات:
        rho: مصفوفة الكثافة 4x4 (معقدة).
    """
    # TODO: تنفيذ خوارزمية الاستعادة الخطية
    # Placeholder logic
    print("تحذير: استخدم الدالة التجريبية `calculate_fidelity_from_angles` للبيانات السريعة.")
    return np.eye(4)/4

def calculate_fidelity(rho_exp, psi_target=None):
    """
    تحسب إخلاص F = ⟨ψ_target|ρ_exp|ψ_target⟩.
    

    المدخلات:
        rho_exp: مصفوفة الكثافة المقاسة.
        psi_target: متجه الحالة المستهدفة. الافتراضي |Ψ⁻⟩.
    
    المخرجات:
        fidelity: قيمة الإخلاص بين 0 و 1.
    """
    if psi_target is None:
        # حالة بل |Ψ⁻⟩ = (1/√2)(|01⟩ - |10⟩)
        psi_target = np.array([0, 1/np.sqrt(2), -1/np.sqrt(2), 0])
    
    # حساب الإخلاص: F = ⟨ψ|ρ|ψ⟩
    fidelity = np.real(psi_target.conj().T @ rho_exp @ psi_target)
    return fidelity

# دالة مساعدة لقياسات سريعة (بدون QST كاملة)
def calculate_fidelity_from_angles(vis_h, vis_v, phase_offset):
    """
    تقدير سريع للإخلاص من قياسات الرؤية (Visibility) والطور.
    مفيد للمراقبة المستمرة والتحكم.
    """
    # معادلة تقريبية للإخلاص من بارامترات التشابك
    F_est = 0.5 * (1 + (vis_h + vis_v)/2 * np.cos(phase_offset))
    return F_est

if __name__ == "__main__":
    # مثال اختباري
    print("اختبار أدوات حساب الإخلاص...")
    test_counts = {'HH': 120, 'HV': 15, 'VH': 10, 'VV': 130, ...} # بيانات وهمية
    rho_test = calculate_density_matrix(test_counts)
    F_test = calculate_fidelity(rho_test)
    print(f"الإخلاص المقدر (من بيانات وهمية): {F_test:.4f}")
```
