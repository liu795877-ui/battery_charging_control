"""以 UTF-8 编码执行并原地保存项目 Notebook。

Windows 中文区域设置可能让 ``jupyter execute`` 按 GBK 读取文件，导致中文
Notebook 在启动内核之前失败。这个小工具显式使用 UTF-8，并把内核工作目录
固定到项目根目录，使 Notebook 中的相对路径保持一致。
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
import sys

import nbformat
from nbclient import NotebookClient


def main() -> None:
    # pyzmq 在 Windows 的 Proactor 循环上会额外创建兼容线程；显式选择
    # Selector 循环可让 Notebook 内核启动更安静、行为也更接近其他平台。
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("notebook", type=Path, help="待执行的 .ipynb 文件")
    parser.add_argument("--timeout", type=int, default=120, help="单个单元格超时秒数")
    args = parser.parse_args()

    notebook_path = args.notebook.resolve()
    project_root = Path(__file__).resolve().parents[1]
    with notebook_path.open("r", encoding="utf-8") as stream:
        notebook = nbformat.read(stream, as_version=4)

    client = NotebookClient(
        notebook,
        timeout=args.timeout,
        kernel_name="python3",
        resources={"metadata": {"path": str(project_root)}},
    )
    client.execute()

    with notebook_path.open("w", encoding="utf-8") as stream:
        nbformat.write(notebook, stream)


if __name__ == "__main__":
    main()
