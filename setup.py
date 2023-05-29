from setuptools import setup, find_packages

setup(
    name='ImageProcessing',  # パッケージ名（pip listで表示される）
    version="0.0.1",  # バージョン
    description="to process image",  # 説明
    author='toshiko',  # 作者名
    packages=find_packages(),  # 使うモジュール一覧を指定する
    license='toshiko'  # ライセンス
)