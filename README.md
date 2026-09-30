# cjwystz.github.io — Chen Jiawei 个人学术主页

基于 [al-folio](https://github.com/alshedivat/al-folio)(Jekyll 学术主题)构建的个人主页,支持论文列表、博客、项目、动态、CV 与 GitHub 仓库展示,可在 GitHub Pages 上免费部署并长期更新。

## 一、部署(只需一次,约 5 分钟)

1. 在 GitHub 新建公开仓库,名字必须是 **`cjwystz.github.io`**(不要勾选 "Add a README")。
2. 把本目录所有文件上传到该仓库根目录(网页上传或 git push 均可)。
3. 仓库 **Settings → Pages → Build and deployment → Source** 选择 **`GitHub Actions`**。
4. 仓库 **Actions** 页会自动运行 `Deploy site` 工作流,1–2 分钟后访问 **`https://cjwystz.github.io`** 即可。

> 仓库里已包含 `.github/workflows/deploy.yml`(自动构建部署)与 `render-cv.yml`(自动用 RenderCV 生成 CV PDF),无需手动配置。

## 二、日常更新(每个都是"改一个文件 + push")

| 想做什么 | 改哪里 |
| --- | --- |
| 发一篇新博客/论文笔记 | 在 `_posts/` 新建 `YYYY-MM-DD-标题.md`,仿照 `_posts/2025-11-15-rag-optimized-t2i.md` 写即可 |
| 新增/修改论文 | 编辑 `_bibliography/papers.bib`,加一条 BibTeX;标 `selected = {true}` 会出现在首页 |
| 更新动态(新闻) | 编辑或新增 `_news/xxx.md` |
| 更新 CV | 编辑 `_data/cv.yml`(推送到 main 后 GitHub Actions 自动渲染新版 CV 页) |
| 改自我介绍/联系方式 | 编辑 `_pages/about.md` |
| 改社交链接 | 编辑 `_data/socials.yml` |
| 改 GitHub 仓库展示 | 编辑 `_data/repositories.yml` |
| 增加项目卡片 | 在 `_projects/` 新建 `.md`(front matter 里写 `category: research` 或 `open-source`) |

## 三、目录速览

```
_pages/          页面(about / publications / blog / projects / cv / news / repositories / 404)
_posts/          博客文章(长期更新的地方)
_news/           首页动态
_projects/       项目卡片
_bibliography/   论文 BibTeX(自动生成 Publications 页)
_data/           cv.yml(简历)、socials.yml(社交)、repositories.yml(GitHub 仓库)
assets/img/      头像 prof_pic.jpg、favicon.svg
_config.yml      站点主配置(站点名、URL、语言、暗色模式等)
```

## 四、自定义提示

- **暗色/浅色切换**:右上角太阳/月亮按钮,默认开启(`_config.yml` 里 `enable_darkmode: true`)。
- **论文状态标签**:bib 条目的 `abbr` 字段显示会议缩写,`html`/`code` 字段自动生成 arXiv/代码链接。
- **更换头像**:直接覆盖 `assets/img/prof_pic.jpg`(建议正方形,300×300 以上)。
- 本仓库 LICENSE 遵循模板上游 al-folio 的 MIT 许可;内容版权归作者本人。
