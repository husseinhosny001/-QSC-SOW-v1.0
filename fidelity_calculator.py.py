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
خوارزمية التعظيم الاحتمالي (MLE) لاستعادة الحالة الكمومية

ملف: /02_SIMULATION_CODE/utilities/tomography_mle.py

```python
"""
خوارزمية RρR للتعظيم الاحتمالي (MLE) لاستعادة الحالة الكمومية.
مبنية على: "Maximum-likelihood estimation of quantum processes" by J. Řeháček et al.
معدلة للبيانات الضوئية مع تصحيح الخلفية والكفاءة.

الإصدار: 1.0.0
تاريخ التعديل: ١٧ ربيع الأول ١٤٤٦ هـ
"""
import numpy as np
import scipy.linalg as la
from scipy.optimize import minimize
from typing import Dict, Tuple, List

# ============================================================================
# الجزء ١: دوال مساعدة أساسية
# ============================================================================

def pauli_matrices() -> List[np.ndarray]:
    """تعيد مصفوفات باولي القياسية I, X, Y, Z."""
    I = np.array([[1, 0], [0, 1]], dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    return [I, X, Y, Z]

def create_bell_state(state: str = 'psi_minus') -> np.ndarray:
    """إنشاء حالة بِلّ متشابكة."""
    if state == 'psi_minus':
        # |Ψ⁻⟩ = (|01⟩ - |10⟩)/√2
        return np.array([0, 1/np.sqrt(2), -1/np.sqrt(2), 0], dtype=complex)
    elif state == 'psi_plus':
        # |Ψ⁺⟩ = (|01⟩ + |10⟩)/√2
        return np.array([0, 1/np.sqrt(2), 1/np.sqrt(2), 0], dtype=complex)
    elif state == 'phi_plus':
        # |Φ⁺⟩ = (|00⟩ + |11⟩)/√2
        return np.array([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)], dtype=complex)
    else:
        raise ValueError(f"Unknown Bell state: {state}")

def density_matrix_to_vector(rho: np.ndarray) -> np.ndarray:
    """تحويل مصفوفة الكثافة إلى متجه بارامترات حقيقية (للتحسين)."""
    # تمثيل ρ كـ ρ = (I + ∑ᵢ cᵢ σᵢ)/4 في قاعدة المنتج التنسوري لمصفوفات باولي
    # العدد ١٥ بارامتر حقيقي (لأن ρ هيرميتية وأثرها ١)
    paulis = pauli_matrices()
    params = []
    
    for i in range(4):
        for j in range(4):
            if i == 0 and j == 0:
                continue  # حذف العنصر I⊗I لأنه ثابت (لتلبية أثر=١)
            P = np.kron(paulis[i], paulis[j])
            params.append(np.real(np.trace(P @ rho)))
    
    return np.array(params)

def vector_to_density_matrix(params: np.ndarray) -> np.ndarray:
    """تحويل متجه البارامترات إلى مصفوفة كثافة (مع فرض الإيجابية)."""
    paulis = pauli_matrices()
    d = 4  # بعد فضاء هيلبرت
    rho = np.eye(d, dtype=complex) / d  # البدء بالمختلط الكلي
    
    idx = 0
    for i in range(4):
        for j in range(4):
            if i == 0 and j == 0:
                continue
            P = np.kron(paulis[i], paulis[j])
            rho += params[idx] * P / 4
            idx += 1
    
    # فرض الإيجابية وأثر=١ باستخدام تحويل إلى القيم الذاتية
    eigvals, eigvecs = la.eigh(rho)
    eigvals = np.maximum(eigvals, 1e-10)  # فرض الإيجابية
    eigvals = eigvals / np.sum(eigvals)   # تطبيع الأثر إلى ١
    
    # إعادة بناء ρ
    rho_positive = eigvecs @ np.diag(eigvals) @ eigvecs.conj().T
    return rho_positive

# ============================================================================
# الجزء ٢: معالجة البيانات وإعداد مشغلي القياس
# ============================================================================

def create_measurement_operators(basis_a: str, basis_b: str) -> Dict[str, np.ndarray]:
    """
    إنشاء مشغلي القياس لتركيبات الاستقطاب المحددة.
    
    المدخلات:
        basis_a, basis_b: واحدة من {'I', 'X', 'Y', 'Z'} لتحديد قاعدة القياس
        لـ Alice و Bob على التوالي.
    
    المخرجات:
        قاموس برمز الحالة ('HH', 'HV', 'VH', 'VV') كمفتاح ومشغل القياس
        المقابل كقيمة.
    """
    # متجهات استقطاب القاعدة الحسابية
    H = np.array([1, 0], dtype=complex)  |0⟩
    V = np.array([0, 1], dtype=complex)  |1⟩
    
    # مصفوفات الدوران للقواعد المختلفة
    rot_matrices = {
        'I': np.eye(2),  # لا دوران
        'X': np.array([[1, 1], [1, -1]], dtype=complex)/np.sqrt(2),  # قاعدة ±45°
        'Y': np.array([[1, 1j], [1, -1j]], dtype=complex)/np.sqrt(2),  # قاعدة دائري
        'Z': np.eye(2)   # قاعدة H/V
    }
    
    U_a = rot_matrices.get(basis_a, rot_matrices['Z'])
    U_b = rot_matrices.get(basis_b, rot_matrices['Z'])
    
    # متجهات الحالة بعد الدوران
    states_a = {
        'H': U_a @ H,
        'V': U_a @ V
    }
    states_b = {
        'H': U_b @ H,
        'V': U_b @ V
    }
    
    # بناء مشغلي القياس كمشغلات إسقاط على الحالات ثنائية الفوتون
    operators = {}
    for out_a in ['H', 'V']:
        for out_b in ['H', 'V']:
            key = out_a + out_b
            psi_a = states_a[out_a].reshape(-1, 1)
            psi_b = states_b[out_b].reshape(-1, 1)
            
            # حالة ثنائية الفوتون
            psi_joint = np.kron(psi_a, psi_b)
            # مشغل القياس (إسقاط)
            operators[key] = psi_joint @ psi_joint.conj().T
    
    return operators

def load_experimental_data(filepath: str) -> Dict[str, Dict[str, float]]:
    """
    تحميل البيانات التجريبية من ملف CSV.
    
    تنسيق الملف المتوقع:
        setting, N_HH, N_HV, N_VH, N_VV, total_time
        "ZZ", 12000, 1500, 1600, 11900, 10.0
        "ZX", 6500, 7000, 6800, 6200, 10.0
        ...
    """
    import csv
    data = {}
    
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            setting = row['setting']
            data[setting] = {
                'HH': float(row['N_HH']),
                'HV': float(row['N_HV']),
                'VH': float(row['N_VH']),
                'VV': float(row['N_VV']),
                'time': float(row['total_time'])
            }
    
    return data

# ============================================================================
# الجزء ٣: خوارزمية RρR الأساسية
# ============================================================================

def rho_mle_rhr(data: Dict[str, Dict[str, float]], 
                max_iter: int = 500, 
                tol: float = 1e-8,
                rho_init: np.ndarray = None) -> Tuple[np.ndarray, Dict]:
    """
    خوارزمية RρR للتعظيم الاحتمالي.
    
    المدخلات:
        data: البيانات التجريبية بالتنسيق المعرفة في load_experimental_data
        max_iter: الحد الأقصى للتكرارات
        tol: حد التقارب
        rho_init: تخمين أولي لـ ρ (افتراضي: حالة مختلطة كاملة)
    
    المخرجات:
        rho_optimal: مصفوفة الكثافة المثلى
        info: قاموس يحتوي معلومات عن عملية التحسين
    """
    # قائمة إعدادات القياس الـ ١٦
    settings = ['II', 'ZI', 'IZ', 'ZZ', 'XI', 'YI', 'XZ', 'YZ', 
                'IX', 'IY', 'ZX', 'ZY', 'XX', 'XY', 'YX', 'YY']
    
    # تهيئة ρ (مختلط كامل إذا لم يُحدد)
    if rho_init is None:
        rho = np.eye(4, dtype=complex) / 4
    else:
        rho = rho_init.copy()
    
    # معايرة مشغلي القياس لجميع الإعدادات
    measurement_ops = {}
    for setting in settings:
        basis_a, basis_b = setting[0], setting[1]
        measurement_ops[setting] = create_measurement_operators(basis_a, basis_b)
    
    # متغيرات لتتبع التقارب
    log_likelihood_history = []
    fidelity_history = []
    
    # حالة بِلّ المستهدفة
    psi_target = create_bell_state('psi_minus')
    rho_target = psi_target.reshape(-1, 1) @ psi_target.conj().reshape(1, -1)
    
    # الخوارزمية التكرارية RρR
    for iteration in range(max_iter):
        # حساب مصفوفة R
        R = np.zeros((4, 4), dtype=complex)
        
        for setting in settings:
            if setting not in data:
                continue
                
            total_counts = sum(data[setting][outcome] for outcome in ['HH', 'HV', 'VH', 'VV'])
            if total_counts == 0:
                continue
            
            for outcome in ['HH', 'HV', 'VH', 'VV']:
                n = data[setting][outcome]
                if n == 0:
                    continue
                    
                M = measurement_ops[setting][outcome]
                p = np.real(np.trace(M @ rho))
                
                if p > 1e-12:  # تجنب القسمة على الصفر
                    weight = n / p
                    R += weight * M
        
        # خطوة التحديث: ρ_{k+1} = N * (R ρ_k R)
        rho_new = R @ rho @ R
        
        # تطبيع للحفاظ على أثر=١
        rho_new = rho_new / np.trace(rho_new)
        
        # فرض الإيجابية (عن طريق إسقاط القيم الذاتية السالبة)
        eigvals, eigvecs = la.eigh(rho_new)
        eigvals = np.maximum(eigvals, 0)
        eigvals = eigvals / np.sum(eigvals)  # تطبيع
        
        rho_new = eigvecs @ np.diag(eigvals) @ eigvecs.conj().T
        
        # حساب دالة اللوغاريتم الاحتمالي
        log_likelihood = 0
        for setting in settings:
            if setting not in data:
                continue
            for outcome in ['HH', 'HV', 'VH', 'VV']:
                n = data[setting][outcome]
                if n == 0:
                    continue
                M = measurement_ops[setting][outcome]
                p = np.real(np.trace(M @ rho_new))
                if p > 1e-12:
                    log_likelihood += n * np.log(p)
        
        log_likelihood_history.append(log_likelihood)
        
        # حساب الإخلاص الحالي
        fidelity = np.real(np.trace(rho_target @ rho_new))
        fidelity_history.append(fidelity)
        
        # التحقق من التقارب
        if iteration > 0:
            delta_rho = la.norm(rho_new - rho, 'fro')
            delta_ll = abs(log_likelihood - log_likelihood_history[-2])
            
            if delta_rho < tol and delta_ll < tol:
                print(f"التقارب بعد {iteration+1} تكرارات")
                break
        
        # تحديث ρ للجولة التالية
        rho = rho_new
    
    # تجميع معلومات النتائج
    info = {
        'iterations': iteration + 1,
        'log_likelihood': log_likelihood_history,
        'fidelity_history': fidelity_history,
        'final_fidelity': fidelity_history[-1],
        'purity': np.real(np.trace(rho @ rho)),
        'eigenvalues': np.linalg.eigvalsh(rho)
    }
    
    return rho, info

# ============================================================================
# الجزء ٤: دوال حساب مؤشرات الجودة
# ============================================================================

def compute_concurrence(rho: np.ndarray) -> float:
    """
    حساب درجة التشابك (Concurrence) لمصفوفة كثافة ثنائية الكيوبت.
    
    Concurrence = max(0, λ₁ - λ₂ - λ₃ - λ₄)
    حيث λᵢ هي الجذور التربيعية للقيم الذاتية لـ ρ(σ_y⊗σ_y)ρ*(σ_y⊗σ_y)
    بترتيب تنازلي.
    """
    # مصفوفة σ_y ⊗ σ_y
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    YY = np.kron(Y, Y)
    
    # حساب R = ρ(σ_y⊗σ_y)ρ*(σ_y⊗σ_y)
    R = rho @ YY @ rho.conj() @ YY
    
    # القيم الذاتية لـ R
    eigvals = np.linalg.eigvals(R)
    eigvals = np.sqrt(np.abs(eigvals))  # الجذور التربيعية للقيم المطلقة
    eigvals.sort()  # ترتيب تصاعدي
    
    # λ₁ - λ₂ - λ₃ - λ₄ (بعد عكس الترتيب)
    concurrence = eigvals[-1] - eigvals[-2] - eigvals[-3] - eigvals[-4]
    
    return max(0, np.real(concurrence))

def compute_entropy(rho: np.ndarray) -> float:
    """حجم إنتروبيا فون نيومان: S = -Tr(ρ log₂ ρ)."""
    eigvals = np.linalg.eigvalsh(rho)
    # تجنب log(0) بإزالة القيم الصغيرة جدًا
    eigvals = eigvals[eigvals > 1e-12]
    entropy = -np.sum(eigvals * np.log2(eigvals))
    return np.real(entropy)

def compute_uncertainty_bootstrap(rho: np.ndarray, data: Dict, 
                                  n_resamples: int = 1000) -> Dict[str, float]:
    """
    حساب عدم اليقين باستخدام طريقة إعادة العينة الإحصائية (Bootstrap).
    
    تعيد قاموسًا يحتوي على:
        - fidelity_mean: متوسط الإخلاص
        - fidelity_std: الانحراف المعياري للإخلاص
        - concurrence_mean: متوسط التشابك
        - concurrence_std: الانحراف المعياري للتشابك
    """
    # هذه وظيفة مكلفة حسابيًا - تستخدم للتحليل النهائي فقط
    import warnings
    warnings.filterwarnings('ignore')
    
    settings = list(data.keys())
    fidelities = []
    concurrences = []
    
    psi_target = create_bell_state('psi_minus')
    rho_target = psi_target.reshape(-1, 1) @ psi_target.conj().reshape(1, -1)
    
    for _ in range(n_resamples):
        # إنشاء عينة إعادة بوتستراب
        resampled_data = {}
        
        for setting in settings:
            # إعادة أخذ العينات متعددة الحدود من العدود
            total_counts = int(sum(data[setting][outcome] for outcome in ['HH', 'HV', 'VH', 'VV']))
            
            if total_counts == 0:
                continue
            
            # توليد عدود جديدة من التوزيع متعدد الحدود
            probs = [data[setting]['HH']/total_counts, 
                    data[setting]['HV']/total_counts,
                    data[setting]['VH']/total_counts, 
                    data[setting]['VV']/total_counts]
            
            new_counts = np.random.multinomial(total_counts, probs)
            
            resampled_data[setting] = {
                'HH': float(new_counts[0]),
                'HV': float(new_counts[1]),
                'VH': float(new_counts[2]),
                'VV': float(new_counts[3]),
                'time': data[setting]['time']
            }
        
        # استعادة ρ من البيانات المعادة أخذ العينات
        try:
            rho_resampled, _ = rho_mle_rhr(resampled_data, max_iter=100, tol=1e-6)
            
            # حساب مقاييس الجودة
            fidelity = np.real(np.trace(rho_target @ rho_resampled))
            concurrence = compute_concurrence(rho_resampled)
            
            fidelities.append(fidelity)
            concurrences.append(concurrence)
        except:
            continue
    
    # حساب الإحصاءات
    results = {
        'fidelity_mean': float(np.mean(fidelities)),
        'fidelity_std': float(np.std(fidelities)),
        'concurrence_mean': float(np.mean(concurrences)),
        'concurrence_std': float(np.std(concurrences)),
        'n_successful_resamples': len(fidelities)
    }
    
    return results

# ============================================================================
# الجزء ٥: مثال تشغيلي واختبار
# ============================================================================

def generate_simulated_data(fidelity: float = 0.99, 
                           total_counts: int = 100000) -> Dict:
    """
    توليد بيانات محاكاة لحالة بِلّ مع ضوضاء.
    
    المدخلات:
        fidelity: الإخلاص المطلوب للحالة
        total_counts: إجمالي عدد العدود
    
    المخرجات:
        بيانات محاكاة بنفس تنسيق البيانات التجريبية
    """
    # حالة بِلّ المثالية مع ضوضاء
    psi_ideal = create_bell_state('psi_minus')
    rho_ideal = psi_ideal.reshape(-1, 1) @ psi_ideal.conj().reshape(1, -1)
    
    # حالة مختلطة كاملة
    rho_mixed = np.eye(4, dtype=complex) / 4
    
    # حالة مع الإخلاص المطلوب
    rho = fidelity * rho_ideal + (1 - fidelity) * rho_mixed
    rho = rho / np.trace(rho)  # تأكيد تطبيع الأثر
    
    # توليد البيانات لجميع الإعدادات الـ ١٦
    settings = ['II', 'ZI', 'IZ', 'ZZ', 'XI', 'YI', 'XZ', 'YZ', 
                'IX', 'IY', 'ZX', 'ZY', 'XX', 'XY', 'YX', 'YY']
    
    data = {}
    
    for setting in settings:
        basis_a, basis_b = setting[0], setting[1]
        measurement_ops = create_measurement_operators(basis_a, basis_b)
        
        # توزيع العدود بناءً على الاحتمالات النظرية
        probs = {}
        total_prob = 0
        
        for outcome in ['HH', 'HV', 'VH', 'VV']:
            prob = np.real(np.trace(measurement_ops[outcome] @ rho))
            probs[outcome] = prob
            total_prob += prob
        
        # تطبيع الاحتمالات وتوليد العدود
        counts = {}
        remaining_counts = total_counts // len(settings)
        
        for outcome in ['HH', 'HV', 'VH', 'VV']:
            expected = probs[outcome] / total_prob * remaining_counts
            # إضافة ضوغة بواسونية لمحاكاة الضوضاء الإحصائية
            counts[outcome] = float(np.random.poisson(expected))
        
        data[setting] = {
            **counts,
            'time': 10.0  # زمن تجميع ثابت للمحاكاة
        }
    
    return data

def run_example():
    """مثال تشغيلي كامل للخوارزمية."""
    print("=" * 60)
    print("اختبار خوارزمية MLE لاستعادة الحالة الكمومية")
    print("=" * 60)
    
    # ١. توليد بيانات محاكاة
    print("١. توليد بيانات محاكاة مع إخلاص ٠.٩٩...")
    simulated_data = generate_simulated_data(fidelity=0.99, total_counts=50000)
    
    # ٢. استعادة الحالة باستخدام MLE
    print("٢. تشغيل خوارزمية RρR...")
    rho_estimated, info = rho_mle_rhr(simulated_data, max_iter=200, tol=1e-8)
    
    # ٣. حساب مؤشرات الجودة
    print("٣. حساب مؤشرات الجودة...")
    psi_target = create_bell_state('psi_minus')
    rho_target = psi_target.reshape(-1, 1) @ psi_target.conj().reshape(1, -1)
    
    fidelity = np.real(np.trace(rho_target @ rho_estimated))
    concurrence = compute_concurrence(rho_estimated)
    purity = np.real(np.trace(rho_estimated @ rho_estimated))
    entropy = compute_entropy(rho_estimated)
    
    # ٤. عرض النتائج
    print("\n" + "=" * 60)
    print("النتائج:")
    print("=" * 60)
    print(f"عدد التكرارات: {info['iterations']}")
    print(f"الإخلاص (Fidelity): {fidelity:.6f}")
    print(f"درجة التشابك (Concurrence): {concurrence:.6f}")
    print(f"النقاء (Purity): {purity:.6f}")
    print(f"الإنتروبيا (Entropy): {entropy:.6f}")
    print(f"القيم الذاتية لـ ρ: {np.real(info['eigenvalues'])}")
    
    # ٥. حساب عدم اليقين (عينة صغيرة للسرعة)
    print("\nحساب عدم اليقين (Bootstrap مع ١٠٠ عينة)...")
    uncertainty = compute_uncertainty_bootstrap(rho_estimated, simulated_data, n_resamples=100)
    
    print(f"متوسط الإخلاص: {uncertainty['fidelity_mean']:.6f} ± {uncertainty['fidelity_std']:.6f}")
    print(f"متوسط التشابك: {uncertainty['concurrence_mean']:.6f} ± {uncertainty['concurrence_std']:.6f}")
    
    return rho_estimated, info, uncertainty

if __name__ == "__main__":
    # عند التشغيل المباشر، نفذ المثال التوضيحي
    rho_est, info, uncert = run_example()
    
    # حفظ النتائج (كمثال)
    np.save('rho_estimated.npy', rho_est)
    print("\nتم حفظ مصفوفة الكثافة المقدرة في 'rho_estimated.npy'")
```

ملاحظات تقنية هامة للاستخدام:

١. التثبيت والاعتماديات:

```bash
pip install numpy scipy matplotlib
```

٢. استخدام مع البيانات التجريبية:

```python
# استيراد المكتبة
from tomography_mle import load_experimental_data, rho_mle_rhr, compute_uncertainty_bootstrap

# ١. تحميل البيانات
data = load_experimental_data('path/to/your/experimental_data.csv')

# ٢. استعادة الحالة
rho, info = rho_mle_rhr(data, max_iter=500, tol=1e-8)

# ٣. حساب عدم اليقين
uncertainty = compute_uncertainty_bootstrap(rho, data, n_resamples=1000)

# ٤. طباعة النتائج
print(f"الإخلاص: {info['final_fidelity']:.4f} ± {uncertainty['fidelity_std']:.4f}")
```

٣. نصائح للتحسين العددي:

1. معالجة البيانات المتناثرة: إذا كانت بعض العدود صفرًا، أضف إزاحة صغيرة (مثل +1) لتجنب log(0).
2. التسريع: استخدم numba أو JAX لتسريع الحلقات التكرارية إذا كانت البيانات كبيرة.
3. الاستقرار: تأكد من أن rho_init قريبة من الحقيقة (يمكن استخدام الاستعادة الخطية كتخمين أولي).

٤. التحقق من الصحة:

يحتوي الكود على مثال داخلي (run_example()) يولد بيانات محاكاة ويختبر الخوارزمية. استخدمه للتحقق من صح
