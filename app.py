import os
import re
import time
import json
import hashlib
import datetime
import pandas as pd
import streamlit as st
from Bio import Entrez

st.set_page_config(
    page_title="MetaMate",
    page_icon="🧪",
    layout="wide"
)

# ================= 配置 =================
DATA_DIR = "meta_project"
os.makedirs(DATA_DIR, exist_ok=True)

ENTREZ_EMAIL = os.getenv("ENTREZ_EMAIL", "your_email@example.com")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

Entrez.email = ENTREZ_EMAIL

# ================= Session State =================
if "project" not in st.session_state:
    st.session_state.project = {
        "name": "",
        "pico": "",
        "inclusion": "",
        "exclusion": "",
        "databases": ["PubMed"],
        "search_date": "",
        "search_query": "",
    }

if "papers" not in st.session_state:
    st.session_state.papers = []

if "screened" not in st.session_state:
    st.session_state.screened = []

if "meta_data" not in st.session_state:
    st.session_state.meta_data = None

if "logs" not in st.session_state:
    st.session_state.logs = []


# ================= 工具函数 =================

def add_log(action: str, detail: str = ""):
    entry = {
        "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "action": action,
        "detail": detail,
    }
    st.session_state.logs.append(entry)


def normalize_title(title: str) -> str:
    title = title.lower()
    title = re.sub(r"[^\w\s]", "", title)
    title = re.sub(r"\s+", " ", title).strip()
    return title


def deduplicate(papers: list[dict]) -> list[dict]:
    seen_dois = set()
    seen_titles = set()
    unique = []
    for p in papers:
        doi = str(p.get("doi", "")).lower().strip()
        title = normalize_title(p.get("title", ""))
        if doi and doi in seen_dois:
            continue
        if title in seen_titles:
            continue
        if doi:
            seen_dois.add(doi)
        seen_titles.add(title)
        unique.append(p)
    return unique


def search_pubmed(query: str, max_results: int = 200) -> list[dict]:
    try:
        handle = Entrez.esearch(
            db="pubmed",
            term=query,
            retmax=min(max_results, 10000),
            usehistory="y",
            sort="relevance",
        )
        results = Entrez.read(handle)
        handle.close()

        total = int(results["Count"])
        webenv = results["WebEnv"]
        query_key = results["QueryKey"]

        batch_size = 500
        papers = []
        for start in range(0, min(total, max_results), batch_size):
            handle = Entrez.efetch(
                db="pubmed",
                rettype="abstract",
                retmode="xml",
                retstart=start,
                retmax=batch_size,
                webenv=webenv,
                query_key=query_key,
            )
            records = Entrez.read(handle)
            handle.close()

            for article in records.get("PubmedArticle", []):
                pmid = article["MedlineCitation"]["PMID"]
                art = article["MedlineCitation"]["Article"]

                title = art.get("ArticleTitle", "")
                abstract = ""
                if art.get("Abstract"):
                    for sec in art["Abstract"].get("AbstractText", []):
                        abstract += str(sec) + " "

                authors = []
                if art.get("AuthorList"):
                    for a in art["AuthorList"]:
                        last = a.get("LastName", "")
                        fore = a.get("ForeName", "")
                        if last:
                            authors.append(f"{last} {fore}".strip())

                journal = art.get("Journal", {}).get("Title", "")
                pub_date = ""
                try:
                    pd_obj = article["MedlineCitation"]["Article"]["Journal"]["JournalIssue"]["PubDate"]
                    pub_date = f"{pd_obj.get('Year', '')}-{pd_obj.get('Month', '')}-{pd_obj.get('Day', '')}"
                except Exception:
                    pub_date = ""

                dois = article.get("PubmedData", {}).get("ArticleIdList", [])
                doi = ""
                for d in dois:
                    if d.attributes.get("IdType") == "doi":
                        doi = str(d)
                        break

                papers.append(
                    {
                        "pmid": str(pmid),
                        "title": str(title).strip(),
                        "abstract": abstract.strip(),
                        "authors": "; ".join(authors[:5]),
                        "journal": journal,
                        "pub_date": pub_date,
                        "doi": doi,
                        "ai_decision": "",
                        "decision_reason": "",
                        "exclude_reason": "",
                        "reviewer1": "",
                        "reviewer2": "",
                        "final_decision": "",
                    }
                )
            time.sleep(0.3)

        return papers
    except Exception as e:
        st.error(f"PubMed 检索失败：{e}")
        return []


def ai_screen_papers(papers: list[dict], inclusion: str, exclusion: str) -> list[dict]:
    screened = []
    for p in papers:
        text = f"标题：{p['title']}\n摘要：{p['abstract'][:800]}"

        decision = "需人工判断"
        reason = ""

        if OPENAI_API_KEY:
            try:
                from openai import OpenAI

                client = OpenAI(api_key=OPENAI_API_KEY)
                resp = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "你是一个医学系统评价筛选助手。"
                                "根据纳入标准和排除标准，判断文献是否可能纳入。"
                                "输出必须包含：可能纳入/可能排除/需人工判断，并给出简短理由。"
                            ),
                        },
                        {
                            "role": "user",
                            "content": (
                                f"纳入标准：{inclusion}\n"
                                f"排除标准：{exclusion}\n\n"
                                f"文献：\n{text}"
                            ),
                        },
                    ],
                    temperature=0.2,
                )
                decision_text = resp.choices[0].message.content.strip()
                decision = decision_text.split("：")[-1].split("\n")[0].strip() or decision_text[:10]
                reason = decision_text
            except Exception:
                decision = "需人工判断"
                reason = "AI 调用失败"

        screened.append(
            {
                **p,
                "ai_decision": decision,
                "decision_reason": reason,
            }
        )
    return screened


def generate_r_script(df: pd.DataFrame) -> tuple[str, str]:
    csv_path = os.path.join(DATA_DIR, "meta_data.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    r_path = os.path.join(DATA_DIR, "meta_analysis.R")

    script = """
library(meta)

data <- read.csv("meta_data.csv", header = TRUE)

# 字段说明：
# Study  : 研究标识
# TE     : 效应量，如 log(OR)、log(RR)、MD、SMD
# seTE   : 标准误

m <- metagen(
  TE = TE,
  seTE = seTE,
  studlab = Study,
  data = data,
  sm = "OR",
  comb.fixed = FALSE,
  comb.random = TRUE
)

print(m, digits = 3)

pdf("forest_plot.pdf", width = 10, height = 8)
forest(
  m,
  leftlabs = c("Study"),
  rightlabs = c("OR", "95% CI"),
  print.I2 = TRUE,
  print.pval.Q = TRUE,
  fontsize = 10
)
dev.off()

cat("森林图已生成\\n")
"""

    with open(r_path, "w", encoding="utf-8") as f:
        f.write(script)

    return csv_path, r_path


def generate_prisma_draft(papers: list[dict], screened: list[dict]) -> dict:
    total_retrieved = len(papers)
    unique_after_dedup = len(screened)

    included = sum(1 for s in screened if s.get("final_decision") == "纳入")
    excluded = sum(1 for s in screened if s.get("final_decision") == "排除")
    pending = sum(1 for s in screened if s.get("final_decision") in ("", "待定", "需人工判断"))

    exclude_reasons = {}
    for s in screened:
        reason = s.get("exclude_reason", "") or s.get("ai_decision", "")
        if reason:
            exclude_reasons[reason] = exclude_reasons.get(reason, 0) + 1

    return {
        "检索获得文献数": total_retrieved,
        "去重后文献数": unique_after_dedup,
        "最终纳入": included,
        "最终排除": excluded,
        "待定": pending,
        "排除原因分布": exclude_reasons,
    }


# ================= 侧边栏 =================
with st.sidebar:
    st.title("MetaMate")
    st.markdown("Meta 分析工作流辅助工具 · 所有 AI 输出需人工复核")

    project = st.session_state.project

    project["name"] = st.text_input(
        "项目名称",
        value=project["name"] or "肺炎诱发心肌病系统评价"
    )
    project["pico"] = st.text_area(
        "PICO / 研究问题",
        value=project["pico"]
        or "P：肺炎患者；I/E：肺炎或重症肺炎；C：非肺炎或轻症患者；O：心肌损伤、心肌炎、心血管并发症、死亡率。",
        height=80,
    )
    project["inclusion"] = st.text_area(
        "纳入标准",
        value=project["inclusion"]
        or "观察性研究或RCT；报告心肌损伤或心血管并发症相关结局；可提取效应量或原始数据。",
        height=80,
    )
    project["exclusion"] = st.text_area(
        "排除标准",
        value=project["exclusion"]
        or "动物研究、体外研究、病例报告、综述、无法提取数据的文献。",
        height=80,
    )

    st.markdown("### 项目状态")
    st.metric("检索文献数", len(st.session_state.papers))
    st.metric("已筛选文献数", len(st.session_state.screened))

    st.markdown("### 合规提示")
    st.info(
        "AI 仅用于辅助筛选、提取和生成草稿。"
        "纳入排除、数据提取、统计模型和最终结论必须由研究者人工确认。"
    )


# ================= 主页面 =================
st.title("MetaMate · Meta分析工作流")
st.markdown("检索 → 去重 → 初筛 → 数据提取 → 统计复现 → 报告导出")

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["1. 检索与去重", "2. 文献筛选", "3. 数据提取", "4. 统计复现", "5. 报告与日志"]
)

# ---------- 1. 检索与去重 ----------
with tab1:
    st.header("PubMed 检索")

    query = st.text_area(
        "检索式",
        value=st.session_state.project.get("search_query")
        or "(pneumonia OR COVID-19) AND (myocarditis OR myocardial injury OR cardiac complications)",
        height=80,
    )
    max_results = st.slider("最大检索数量", 50, 500, 200)

    if st.button("开始检索"):
        with st.spinner("正在检索 PubMed..."):
            papers = search_pubmed(query, max_results=max_results)
            unique_papers = deduplicate(papers)
            st.session_state.papers = papers
            st.session_state.screened = unique_papers
            st.session_state.project["search_query"] = query
            st.session_state.project["search_date"] = datetime.date.today().isoformat()
            add_log("PubMed 检索", f"检索式：{query}；去重后：{len(unique_papers)} 篇")
            st.success(f"检索完成，去重后共 {len(unique_papers)} 篇文献")

    if st.session_state.papers:
        df = pd.DataFrame(st.session_state.papers)
        st.dataframe(df, use_container_width=True)

        csv_bytes = df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8")
        st.download_button(
            "下载检索结果 CSV",
            csv_bytes,
            file_name="pubmed_results.csv",
            mime="text/csv",
        )


# ---------- 2. 文献筛选 ----------
with tab2:
    st.header("AI 文献初筛")

    inclusion = st.session_state.project["inclusion"]
    exclusion = st.session_state.project["exclusion"]

    if st.button("开始 AI 初筛"):
        if not st.session_state.papers:
            st.warning("请先完成文献检索")
        else:
            with st.spinner("正在进行 AI 初筛..."):
                screened = ai_screen_papers(
                    st.session_state.screened or st.session_state.papers,
                    inclusion,
                    exclusion,
                )
                st.session_state.screened = screened
                add_log("AI 初筛", f"共 {len(screened)} 篇")
                st.success(f"初筛完成，共 {len(screened)} 篇")

    if st.session_state.screened:
        df_screen = pd.DataFrame(st.session_state.screened)

        st.subheader("人工复核表")
        st.markdown(
            "AI 决策仅供参考，请逐条确认纳入/排除/待定，并填写排除原因。"
        )

        for idx, row in df_screen.iterrows():
            with st.expander(
                f"{idx + 1}. {row.get('title', '')[:120]}"
            ):
                st.markdown(f"**标题**：{row.get('title', '')}")
                st.markdown(f"**摘要**：{row.get('abstract', '')[:600]}")
                st.markdown(f"**AI 判断**：{row.get('ai_decision', '')}")
                st.markdown(f"**AI 理由**：{row.get('decision_reason', '')}")

                final = st.selectbox(
                    "最终决策",
                    ["", "纳入", "排除", "待定"],
                    key=f"decision_{idx}",
                )
                exclude_reason = st.text_input(
                    "排除原因",
                    value=row.get("exclude_reason", ""),
                    key=f"exclude_{idx}",
                )

                st.session_state.screened[idx]["final_decision"] = final
                st.session_state.screened[idx]["exclude_reason"] = exclude_reason

        df_final = pd.DataFrame(st.session_state.screened)
        csv_bytes = df_final.to_csv(index=False, encoding="utf-8-sig").encode("utf-8")
        st.download_button(
            "下载筛选结果 CSV",
            csv_bytes,
            file_name="screened_results.csv",
            mime="text/csv",
        )


# ---------- 3. 数据提取 ----------
with tab3:
    st.header("数据提取")

    st.markdown(
        "建议上传包含以下字段的 CSV：Study, TE, seTE。"
        "TE 为效应量，seTE 为标准误。也可上传 PDF 表格进行初步提取。"
    )

    uploaded = st.file_uploader("上传 meta_data.csv", type=["csv"])

    if uploaded:
        df_meta = pd.read_csv(uploaded)
        st.dataframe(df_meta, use_container_width=True)
        st.session_state.meta_data = df_meta
        add_log("数据提取", f"上传 meta_data.csv，共 {len(df_meta)} 行")

    st.subheader("PDF 表格提取")
    pdf_file = st.file_uploader("上传文献 PDF", type=["pdf"])

    if pdf_file:
        try:
            import pdfplumber
            import io

            tables = []
            with pdfplumber.open(io.BytesIO(pdf_file.read())) as pdf:
                for i, page in enumerate(pdf.pages):
                    page_tables = page.extract_tables()
                    for t in page_tables:
                        if t and len(t) > 1:
                            df_table = pd.DataFrame(t[1:], columns=t[0])
                            df_table["source_page"] = i + 1
                            tables.append(df_table)

            if tables:
                st.success(f"共提取到 {len(tables)} 个表格")
                for i, t in enumerate(tables):
                    st.markdown(f"**表格 {i + 1}**")
                    st.dataframe(t, use_container_width=True)
                add_log("PDF 表格提取", f"提取到 {len(tables)} 个表格")
            else:
                st.warning("未提取到表格，请检查 PDF 是否为可解析版本。")
        except Exception as e:
            st.error(f"PDF 解析失败：{e}")


# ---------- 4. 统计复现 ----------
with tab4:
    st.header("生成 R Meta 分析脚本")

    if st.session_state.meta_data is not None:
        csv_path, r_path = generate_r_script(st.session_state.meta_data)

        st.success("已生成 R 脚本和数据文件")
        st.markdown(f"数据文件：`{csv_path}`")
        st.markdown(f"R 脚本：`{r_path}`")

        with open(r_path, "r", encoding="utf-8") as f:
            st.code(f.read(), language="r")

        st.info(
            "请在 RStudio 中安装 `meta` 包，然后运行该脚本生成森林图。"
            "若纳入研究数量足够，可进一步补充漏斗图、敏感性分析和亚组分析。"
        )

        add_log("统计复现", "已生成 R 脚本和 meta_data.csv")
    else:
        st.warning("请先在「数据提取」页面上传 meta_data.csv")


# ---------- 5. 报告与日志 ----------
with tab5:
    st.header("PRISMA 流程图草稿")

    if st.session_state.papers and st.session_state.screened:
        prisma = generate_prisma_draft(st.session_state.papers, st.session_state.screened)
        st.json(prisma, expanded=False)

        st.markdown("### 方法学段落草稿")
        draft = (
            f"本项目检索了 PubMed 数据库，检索式为：{st.session_state.project['search_query']}。"
            f"检索日期为 {st.session_state.project['search_date']}。"
            f"共检索到 {prisma['检索获得文献数']} 篇文献，去重后剩余 {prisma['去重后文献数']} 篇。"
            f"经标题/摘要筛选和全文评估后，最终纳入 {prisma['最终纳入']} 篇研究。"
        )
        st.markdown(draft)

        st.markdown("### 操作日志")
        if st.session_state.logs:
            st.dataframe(
                pd.DataFrame(st.session_state.logs),
                use_container_width=True,
            )
        else:
            st.info("暂无操作日志")
    else:
        st.info("请先完成检索和筛选，再生成报告草稿")