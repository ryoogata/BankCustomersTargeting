# 利用可能なモデル数
ml <- caret::modelLookup()
length(unique(ml$model))
# [1] 213
View(caret::modelLookup())
