# src/utils/prompts.py (Final Optimized Version with Strict Segment-Level Logical Consistency)

# ==============================================================================
#  注意：此文件包含两个独立的、基于节段且逻辑互斥的知识库。
#  请根据您的训练任务（诊断心肌缺血 或 诊断心肌梗死），
#  选择使用 "KNOWLEDGE_BASE_FOR_ISCHEMIA" 或 "KNOWLEDGE_BASE_FOR_INFARCTION"。
# ==============================================================================


# ==============================================================================
#  知识库一：用于心肌缺血早期诊断 (KNOWLEDGE_BASE_FOR_ISCHEMIA)
#  ------------------------------------------------------------------------------
#  核心特征: 聚焦于可逆的功能性异常 (Hypokinesis)。每个节段只有两种互斥状态。
# ==============================================================================

# --- ISCHEMIA: A2C切面专属知识 (6个节段 * 2个状态 = 12条知识) ---
A2C_KNOWLEDGE_ISCHEMIA = [
    # Basal Anterior Segment
    "The basal anterior segment exhibits normokinesis with normal systolic thickening.",
    "The basal anterior segment exhibits hypokinesis, consistent with ischemia.",
    # Mid Anterior Segment
    "The mid anterior segment exhibits normokinesis.",
    "The mid anterior segment exhibits hypokinesis.",
    # Apical Anterior Segment
    "The apical anterior segment exhibits normokinesis.",
    "The apical anterior segment exhibits hypokinesis.",
    # Basal Inferior Segment
    "The basal inferior segment exhibits normokinesis.",
    "The basal inferior segment exhibits hypokinesis.",
    # Mid Inferior Segment
    "The mid inferior segment exhibits normokinesis.",
    "The mid inferior segment exhibits hypokinesis.",
    # Apical Inferior Segment
    "The apical inferior segment exhibits normokinesis.",
    "The apical inferior segment exhibits hypokinesis.",
]

# --- ISCHEMIA: A4C切面专属知识 (6个节段 + 1个心尖顶 * 2个状态 = 14条知识) ---
A4C_KNOWLEDGE_ISCHEMIA = [
    # Basal Inferoseptal Segment
    "The basal inferoseptal segment exhibits normokinesis.",
    "The basal inferoseptal segment exhibits hypokinesis.",
    # Mid Inferoseptal Segment
    "The mid inferoseptal segment exhibits normokinesis.",
    "The mid inferoseptal segment exhibits hypokinesis.",
    # Apical Septal Segment
    "The apical septal segment exhibits normokinesis.",
    "The apical septal segment exhibits hypokinesis.",
    # Basal Anterolateral Segment
    "The basal anterolateral segment exhibits normokinesis.",
    "The basal anterolateral segment exhibits hypokinesis.",
    # Mid Anterolateral Segment
    "The mid anterolateral segment exhibits normokinesis.",
    "The mid anterolateral segment exhibits hypokinesis.",
    # Apical Lateral Segment
    "The apical lateral segment exhibits normokinesis.",
    "The apical lateral segment exhibits hypokinesis.",
    # Apical Cap (Segment 17)
    "The apical cap exhibits normal contractility.",
    "The apical cap exhibits hypokinesis.",
]

# --- ISCHEMIA: 最终组合知识库 ---
KNOWLEDGE_BASE_FOR_ISCHEMIA = A2C_KNOWLEDGE_ISCHEMIA + A4C_KNOWLEDGE_ISCHEMIA

# ==============================================================================
#  知识库二：用于心肌梗死诊断 (KNOWLEDGE_BASE_FOR_INFARCTION)
#  ------------------------------------------------------------------------------
#  核心特征: 聚焦于不可逆的结构性和功能性异常 (Akinesis/Dyskinesis/Scarring)。
# ==============================================================================

# --- INFARCTION: A2C切面专属知识 (6个节段 * 2个状态 = 12条知识) ---
A2C_KNOWLEDGE_INFARCTION = [
    # Basal Anterior Segment
    "The basal anterior segment is of normal thickness and demonstrates normokinesis.",
    "The basal anterior segment shows signs of infarction (akinesis or dyskinesis).",
    # Mid Anterior Segment
    "The mid anterior segment is of normal thickness and demonstrates normokinesis.",
    "The mid anterior segment is akinetic and appears thinned and echogenic, suggesting scar.",
    # Apical Anterior Segment
    "The apical anterior segment is of normal thickness and demonstrates normokinesis.",
    "The apical anterior segment is dyskinetic, consistent with aneurysm formation.",
    # Basal Inferior Segment
    "The basal inferior segment is of normal thickness and demonstrates normokinesis.",
    "The basal inferior segment shows signs of infarction (akinesis or dyskinesis).",
    # Mid Inferior Segment
    "The mid inferior segment is of normal thickness and demonstrates normokinesis.",
    "The mid inferior segment is akinetic and appears thinned and echogenic.",
    # Apical Inferior Segment
    "The apical inferior segment is of normal thickness and demonstrates normokinesis.",
    "The apical inferior segment is dyskinetic, consistent with aneurysm formation.",
]

# --- INFARCTION: A4C切面专属知识 (6个节段 + 1个心尖顶 + 2个全局 * 2个状态 = 18条知识) ---
A4C_KNOWLEDGE_INFARCTION = [
    # Basal Inferoseptal Segment
    "The basal inferoseptal segment is of normal thickness and demonstrates normokinesis.",
    "The basal inferoseptal segment shows signs of infarction (akinesis or dyskinesis).",
    # Mid Inferoseptal Segment
    "The mid inferoseptal segment is of normal thickness and demonstrates normokinesis.",
    "The mid inferoseptal segment is akinetic and appears thinned.",
    # Apical Septal Segment
    "The apical septal segment is of normal thickness and demonstrates normokinesis.",
    "The apical septal segment is dyskinetic.",
    # Basal Anterolateral Segment
    "The basal anterolateral segment is of normal thickness and demonstrates normokinesis.",
    "The basal anterolateral segment shows signs of infarction (akinesis or dyskinesis).",
    # Mid Anterolateral Segment
    "The mid anterolateral segment is of normal thickness and demonstrates normokinesis.",
    "The mid anterolateral segment is akinetic and appears scarred.",
    # Apical Lateral Segment
    "The apical lateral segment is of normal thickness and demonstrates normokinesis.",
    "The apical lateral segment is dyskinetic.",
    # Apical Cap (Segment 17) & Complications
    "The apex has a normal conical shape and contractile function.",
    "The apex is damaged by infarction, showing akinesis, aneurysm formation, or an adherent mural thrombus.",
    # LV Remodeling
    "The left ventricle maintains its normal ellipsoid geometry.",
    "The left ventricle appears dilated and spherical, consistent with adverse post-infarction remodeling."
]

# --- INFARCTION: 最终组合知识库 ---
KNOWLEDGE_BASE_FOR_INFARCTION = A2C_KNOWLEDGE_INFARCTION + A4C_KNOWLEDGE_INFARCTION

# ==============================================================================
#  最终知识库选择 (IMPORTANT)
#  ------------------------------------------------------------------------------
#  请在此处选择您想要用于本次训练的知识库。
# ==============================================================================

# --- Option 1: For Myocardial INFARCTION Diagnosis ---
FAEC_KNOWLEDGE_BASE = KNOWLEDGE_BASE_FOR_INFARCTION

# --- Option 2: For Myocardial ISCHEMIA Diagnosis ---
# FAEC_KNOWLEDGE_BASE = KNOWLEDGE_BASE_FOR_ISCHEMIA

# ==============================================================================
#  (兼容性部分)
# ==============================================================================
COMMON_KNOWLEDGE = []

if FAEC_KNOWLEDGE_BASE == KNOWLEDGE_BASE_FOR_INFARCTION:
    A2C_SPECIFIC_KNOWLEDGE = A2C_KNOWLEDGE_INFARCTION
    A4C_SPECIFIC_KNOWLEDGE = A4C_KNOWLEDGE_INFARCTION
else:
    A2C_SPECIFIC_KNOWLEDGE = A2C_KNOWLEDGE_ISCHEMIA
    A4C_SPECIFIC_KNOWLEDGE = A4C_KNOWLEDGE_ISCHEMIA

segment_prompts = {
    "A2C": FAEC_KNOWLEDGE_BASE,
    "A4C": FAEC_KNOWLEDGE_BASE,
    "ALL": FAEC_KNOWLEDGE_BASE
}