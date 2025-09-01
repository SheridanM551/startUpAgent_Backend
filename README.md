### version: v1

### description
combined news RAG (from chrunchBase) and statistic report (from growjo). User can fill in the survey form, and the data will be sent to backend process, returning Json format data: {"statistic": json, "news": {"content": json, "citations": json}}

### 流程
1. user-input
2. send to backend
3. topk company retrieve & calculate statistic
4. topk news retrieve
5. combine news + company + statistic => report

### news
using:
- overall 
    - Industry_Group description
    - Industry_Group plot
- within Industry_Group
    - current_employees plot and desc
    - total_funding plot and desc

get from user_plot, withdraw advisor_main.

### TODO
- add new question(usr desc his company detail)
- adjust all pages style (css)
- update README.md
    - describe Project detail
    - describe code structure
    - add frontend gh page link