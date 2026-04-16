# Define registry options for Q1
q1_options = """
HYDROCEPHALUS/CONGENITAL:
- Aqueduct stenosis
- Dandy-Walker Syndrome
- Arnold-Chiari Syndrome
- Spinal bifida with hydrocephalus
- Congenital hydrocephalus, unspecified
- Arachnoid and porencephalic cysts

INFECTION/INFLAMMATION:
- Intracranial abscess or granuloma
- Extradural or subdural abscess/empyema
- Bacterial meningitis
- Meningitis other than bacterial
- Encephalitis, myelitis, encephalomyelitis
- Other infection

TRAUMA:
- Diffuse brain injury
- Focal brain injury
- Traumatic subarachnoid haemorrhage
- Acute subdural haematoma
- Chronic subdural haematoma
- Unspecified injury to the head

TUMOUR:
Malignant Tumour:
- Primary malignant tumour, supratentorial
- Primary malignant tumour of brain, infratentorial
- Secondary malignant CNS tumour of brain / meninges
- Secondary malignant neoplasm of pituitary gland
- Primary malignant tumour of the pituitary

Benign Tumour:
- Benign supratentorial tumour
- Benign infratentorial tumour
- Benign tumour of cranial nerves
- Benign tumour of meninges, crania
- Benign pituitary region tumours

Unknown Tumour:
- Neoplasm of uncertain or unknown behaviour of brain, supratentorial
- Extradural or subdural abscess / empyema
- Neoplasm of uncertain or unknown behaviour of cerebral meninges
- Neoplasm of uncertain or unknown behaviour of cranial nerves
- Neoplasm of uncertain or unknown behaviour of pituitary gland

CEREBROVASCULAR:
- Aneurysmal subarachnoid haemorrhage
- Non-aneurysmal subarachnoid haemorrhage
- Subarachnoid haemorrhage - unknown cause
- Arteriovenous malformation
- Spontaneous intracerebral haemorrhage
- Spontaneous posterior fossa haemorrhage
- Other non-traumatic intracranial haemorrhage
- Other cerebrovascular disease
- Unruptured intracranial aneurysm

MISCELLANEOUS:
- Idiopathic intracranial hypertension
- Normal pressure hydrocephalus
- Other diagnosis

UNKNOWN
"""

# Define registry options for Q2
q2_options = """
- Yes
- No
"""

# Define registry options for Q4
q4_options = """
- Brain slump
- Catheter fracture
- Catheter migration
- Disconnection
- Distal fracture
- Distal underdrainage
- Dural enhancement
- Infection
- Mechanical failure
- Overdrainage
- Postural headache
- Proximal fracture
- Proximal underdrainage
- Proximal / valve disconnection
- Secondary craniostenosis
- Shunt infection
- Shunt not required
- Slit ventricles
- Subdural haematoma
- Subdural hygroma
- Underdrainage
- Valve / distal disconnection
- Valve underdrainage
- Wound infection only
- Other disconnection
"""

# Define registry options for Q8
q8_options = """
- Yes
- No
"""

# Define registry options for Q9
q9_options = """
- Yes
- No
"""

# Define registry options for Q10
q10_options = """
- Large
- Normal
- Small
"""

# Define registry options for Q11
q11_options = """
- Yes
- No
"""

# Define registry options for Q12
q12_options = """
- Yes
- No
"""

# Define registry options for Q13
q13_options = """
- Yes
- No
"""

# Define registry options for Q18
q18_options = """
- Scrubbed
- Unscrubbed
- Available
- No involvement
"""

# Q23: plain label for prompt template only; schema uses free_text_answer_schema()
q23_options = "Operation title"

# Q25: FIELD label for prompt template; schema is free_text_answer_schema()
q25_options = "Procedure"

# Q26: FIELD label for prompt template; schema is free_text_answer_schema()
q26_options = "Post-op plan"