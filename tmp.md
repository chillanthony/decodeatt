# 检查待推送对象总大小

```bash
git rev-list --objects origin/main..HEAD |
git cat-file --batch-check='%(objecttype) %(objectname) %(objectsize)' |
awk '$1=="blob"{sum+=$3} END{printf "%.1f MB\n",sum/1048576}'
```
