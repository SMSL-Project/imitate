from setuptools import setup, find_packages

setup(
    name='gensim',
    version='0.1.0',
    packages=find_packages(),
    include_package_data=True,
    package_data={"smsl_gensim": ["SMSL_JSON/*.json", "TASK_LIST/*.json"]},
    license=open('LICENSE').read(),
    zip_safe=False,
    description="SMSL imitation-learning backend adapted from GenSim by Lirui Wang et al.",
    author='Yihao Liu',
    author_email='yliu333@jhu.edu',
    keywords=['Large Language Models', 'Simulation', 'Vision Language Grounding', 'Robotics', 'Manipulation'],
)
