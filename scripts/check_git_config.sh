#!/usr/bin/env bash
# ============================================================================
# check_git_config.sh —— git 配置自检 + 常用命令速查
#
# 用法：
#   bash scripts/check_git_config.sh            # 本地检查，不联网
#   bash scripts/check_git_config.sh --network  # 额外测试 GitHub SSH 连通
#
# 说明：
#   1. 在任意目录都可运行；仓库相关检查只在 git 仓库内执行。
#   2. 本脚本只读取配置、不做任何修改；需要修改的地方会打印出对应命令，
#      由你自己决定是否执行。
# ============================================================================

set -u
NETWORK=0
[ "${1:-}" = "--network" ] && NETWORK=1

hr()   { printf '\n========== %s ==========\n' "$1"; }
ok()   { echo "  [OK]  $1"; }
miss() { echo "  [缺失] $1"; }
tip()  { echo "  [提示] $1"; }

# ---------------------------------------------------------------------------
hr "1. git 是否安装"
# ---------------------------------------------------------------------------
if command -v git >/dev/null 2>&1; then
    ok "git $(git --version | awk '{print $3}')，路径 $(command -v git)"
else
    miss "未安装 git。Ubuntu 安装：sudo apt install git"
    exit 1
fi

# ---------------------------------------------------------------------------
hr "2. 全局身份（提交记录的署名）"
# ---------------------------------------------------------------------------
name=$(git config --global --get user.name  || true)
email=$(git config --global --get user.email || true)
if [ -n "$name" ]; then ok "user.name  = $name"; else miss 'user.name 未设置。执行: git config --global user.name "你的名字"'; fi
if [ -n "$email" ]; then ok "user.email = $email"; else miss 'user.email 未设置。执行: git config --global user.email "你的邮箱"'; fi

# ---------------------------------------------------------------------------
hr "3. 常用全局配置（建议值）"
# ---------------------------------------------------------------------------
check_cfg() {  # check_cfg <键> <建议值> <说明>
    local key="$1" suggest="$2" desc="$3" val
    val=$(git config --global --get "$key" || true)
    if [ -n "$val" ]; then
        ok "$key = $val（$desc）"
    else
        miss "$key 未设置（$desc）。建议: git config --global $key $suggest"
    fi
}
check_cfg "init.defaultBranch" "main"      "新建仓库的默认分支名"
check_cfg "pull.rebase"        "true"      "git pull 默认用 rebase，历史更干净"
check_cfg "fetch.prune"        "true"      "fetch 时自动清理远端已删除的分支引用"
check_cfg "core.autocrlf"      "input"     "换行符处理（Linux 建议 input）"

editor=$(git config --global --get core.editor || true)
if [ -n "$editor" ]; then ok "core.editor = $editor"; else tip "core.editor 未设置，默认用 vi。想用 nano: git config --global core.editor nano"; fi

# ---------------------------------------------------------------------------
hr "4. 常用别名（git st 这种短命令就靠它们）"
# ---------------------------------------------------------------------------
found_alias=0
for a in st co br ci lg; do
    v=$(git config --global --get "alias.$a" || true)
    if [ -n "$v" ]; then ok "alias.$a = git $v"; found_alias=1; fi
done
[ "$found_alias" -eq 0 ] && miss "还没有配置任何别名"
tip "没有别名的话，可以执行下面几条一次性配好："
cat <<'ALIAS'
  git config --global alias.st "status -sb"
  git config --global alias.co checkout
  git config --global alias.br "branch -vv"
  git config --global alias.ci commit
  git config --global alias.lg  "log --oneline --graph --decorate --all"
ALIAS

# ---------------------------------------------------------------------------
hr "5. 代理与凭据"
# ---------------------------------------------------------------------------
for k in http.proxy https.proxy; do
    v=$(git config --global --get "$k" || true)
    [ -n "$v" ] && ok "$k = $v" || tip "$k 未设置（直连。需要代理时: git config --global $k http://127.0.0.1:端口）"
done
ch=$(git config --global --get credential.helper || true)
[ -n "$ch" ] && ok "credential.helper = $ch" || tip "credential.helper 未设置（HTTPS 推送时每次要输密码。可执行: git config --global credential.helper store）"

# ---------------------------------------------------------------------------
hr "6. SSH 密钥"
# ---------------------------------------------------------------------------
shopt -s nullglob
pubkeys=(~/.ssh/id_*.pub)
shopt -u nullglob
if [ ${#pubkeys[@]} -gt 0 ]; then
    for p in "${pubkeys[@]}"; do ok "找到公钥 $p"; done
else
    miss "~/.ssh 下没有公钥。生成: ssh-keygen -t ed25519 -C \"$email\"，然后把 *.pub 内容填到代码托管平台"
fi
if [ "$NETWORK" -eq 1 ]; then
    out=$(ssh -o ConnectTimeout=5 -o BatchMode=yes -T git@github.com 2>&1 || true)
    case "$out" in
        *"successfully authenticated"*|*"Hi "*) ok "GitHub SSH 认证通过：$(echo "$out" | head -1)" ;;
        *) warn "GitHub SSH 未通过：$(echo "$out" | head -1)" ;;
    esac
else
    tip "跳过 GitHub 连通测试（加 --network 参数可测）"
fi

# ---------------------------------------------------------------------------
hr "7. 当前仓库状态"
# ---------------------------------------------------------------------------
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    top=$(git rev-parse --show-toplevel)
    branch=$(git branch --show-current)
    ok "仓库根目录: $top"
    ok "当前分支: ${branch:-（无，可能处于 detached HEAD）}"

    n_remote=$(git remote | wc -l)
    if [ "$n_remote" -gt 0 ]; then
        git remote -v | sed 's/^/        /'
    else
        miss "没有配置远程仓库。添加: git remote add origin <仓库地址>"
    fi

    staged=$(git diff --cached --name-only | wc -l)
    unstaged=$(git diff --name-only | wc -l)
    untracked=$(git ls-files --others --exclude-standard | wc -l)
    ok "待提交: 已暂存 $staged 个，未暂存 $unstaged 个，未跟踪 $untracked 个"
    [ $((staged + unstaged + untracked)) -gt 0 ] && tip "用 git st 查看，再决定 git add <文件> 或 git add -p 逐块暂存"

    if upstream=$(git rev-parse --abbrev-ref --symbolic-full-name @{u} 2>/dev/null); then
        ahead=$(git rev-list --count @{u}..HEAD 2>/dev/null || echo "?")
        behind=$(git rev-list --count HEAD..@{u} 2>/dev/null || echo "?")
        ok "跟踪分支: $upstream（本地领先 $ahead，落后 $behind）"
        [ "$ahead" -gt 0 ] && tip "有 $ahead 个提交未推送: git push"
        [ "$behind" -gt 0 ] && tip "远端有 $behind 个新提交: git pull --rebase"
    else
        tip "当前分支没有设置上游。首次推送: git push -u origin $branch"
    fi

    if [ -f "$top/.gitmodules" ]; then
        ok "含子模块（.gitmodules）"
        # 找出已登记为 gitlink 但 .gitmodules 缺少映射的路径
        miss_map=$(comm -23 \
            <(git ls-files --stage | awk '$1==160000{print $4}' | sort) \
            <(git config -f "$top/.gitmodules" --get-regexp '\.path$' | awk '{print $2}' | sort))
        [ -n "$miss_map" ] && miss "以下路径是子模块但 .gitmodules 缺少映射：$(echo $miss_map | tr '\n' ' ')"
        bad=$(git submodule status --recursive 2>/dev/null | grep -c '^[-+U]' || true)
        [ "$bad" -gt 0 ] && tip "有 $bad 个子模块未初始化或版本不一致，执行: git submodule update --init --recursive"
        git submodule status --recursive 2>/dev/null | sed 's/^/        /'
    fi
else
    tip "当前目录不在 git 仓库内，跳过仓库检查"
fi

# ---------------------------------------------------------------------------
hr "8. 常用命令速查"
# ---------------------------------------------------------------------------
cat <<'CHEAT'
【日常】
  git st                       # 状态简报（= status -sb，见第 4 节别名）
  git status                   # 完整状态
  git diff                     # 未暂存的改动
  git diff --staged            # 已暂存、待提交的改动
  git add <文件>               # 暂存指定文件
  git add -p                   # 逐块确认后再暂存（推荐）
  git commit -m "说明"         # 提交
  git commit --amend           # 补充修改到上一次提交（未推送时用）

【看历史】
  git lg                       # 图形化简史（= log --oneline --graph --all）
  git log --oneline -10        # 最近 10 条
  git log -p <文件>            # 某文件的修改历史
  git blame <文件>             # 每行最后是谁改的
  git show <提交号>            # 看某次提交的完整改动

【分支】
  git switch -c <新分支>       # 创建并切换
  git switch <分支>            # 切换
  git branch -vv               # 本地分支及其跟踪关系
  git branch -d <分支>         # 删除已合并分支
  git merge <分支>             # 合并
  git rebase <分支>            # 变基（先确认分支上没有未推送提交）

【远程与同步】
  git fetch --prune            # 拉取远端信息并清理失效引用
  git pull --rebase            # 拉取并用 rebase 合并
  git push                     # 推送
  git push -u origin <分支>    # 首次推送并建立跟踪
  git remote -v                # 查看远程地址

【子模块】
  git clone --recurse-submodules <地址>       # 克隆时带上子模块
  git submodule update --init --recursive     # 初始化/更新子模块
  git submodule status                        # 查看子模块状态
  git submodule add <地址> <路径>             # 添加子模块

【暂存现场】
  git stash                    # 把当前改动先收起来
  git stash list               # 查看收起来的列表
  git stash pop                # 取回最近一次

【找回】
  git reflog                   # 所有 HEAD 移动记录，误操作后在这里找回提交号
  git cherry-pick <提交号>     # 把别的分支上的某次提交摘过来

【慎用（会丢数据或改写历史）】
  git restore <文件>           # 丢弃工作区改动（不可恢复）
  git reset --soft HEAD~1      # 撤销上一次提交，改动保留在暂存区
  git reset --hard HEAD~1      # 撤销提交并丢弃改动（危险）
  git clean -fd                # 删除所有未跟踪文件（危险，先 git clean -nd 预览）
  git push --force             # 强推覆盖远端（危险，先 --force-with-lease）
CHEAT

hr "检查完成"
