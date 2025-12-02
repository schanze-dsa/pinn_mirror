# 合并冲突处理说明

出现 GitHub 页面提示 `src/inp_io/inp_contacts.py` 无法在网页编辑器解决冲突时，可以按以下步骤在本地/命令行完成：

1. **确认预期行为**：该文件已在当前分支移除，功能由 `src/train/attach_ties_bcs.py` 直接从解析后的装配对象读取 Tie/Boundary 信息实现，不再需要单独的 INP 解析器。
2. **获取最新代码**：在本地仓库执行 `git fetch` 并切换到需要合并的分支（例如 `git checkout work`）。
3. **接受删除方案**：运行 `git checkout --theirs src/inp_io/inp_contacts.py`（若目标分支是当前分支则改用 `--ours`），或直接删除该文件 `git rm src/inp_io/inp_contacts.py`，表示以当前分支的“删除”版本为准。
4. **标记解决并提交**：执行 `git add src/inp_io/inp_contacts.py` 然后 `git commit`，再继续合并或推送。

这样可快速消除网页端提示的复杂冲突，并保持代码库使用统一的 Tie/Boundary 挂载实现。
