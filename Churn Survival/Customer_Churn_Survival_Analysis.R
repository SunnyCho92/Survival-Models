library(readr)
library(survminer)
library(survival)
require(GGally)
library(ggplot2)

data <- read_csv('./survival_data.csv')
data$...1 <- NULL

no <- subset(data, channel_way == '눈높이_단독')
min(no$zstd_stdt_01)
max(no$zstd_stdt_01)
min(no$zstd_eddt_01)
max(no$zstd_eddt_01)
no.fit <- survfit(Surv(dur1, status) ~ 1, data=no)

noon <- subset(data, channel == '눈높이')
ng <- survfit(Surv(dur1, status) ~ way, data=noon)
summary(ng)$table
summary(ng, times=c(30, 60, 90, 120, 150, 180, 210))
summary(ng, times=c(30, 60, 90, 120, 150, 180, 210, 240))
survdiff(Surv(dur1, status) ~ way, data=noon)

#시각화
ggsurvplot(ng, pval=TRUE, conf.int = TRUE, xlab="Days", break.x.by = 30, linetype = "strata", surv.median.line = "hv", ggtheme = theme_classic2(base_family ="MaruBuri-Bold"), font.family = "MaruBuri-Bold", palette = c("#FC0000", "#6E888E"), data=noon)

sol <- subset(data, channel == '솔루니')
sg <- survfit(Surv(dur1, status) ~ way, data=sol)
survdiff(Surv(dur1, status) ~ way, data=sol)
sg <- survfit(Surv(dur1, status) ~ way, data=sol)
summary(sg, times=c(30, 60, 90, 120, 150, 180, 210))
summary(sg)$table

#시각화
ggsurvplot(sg, pval=TRUE, conf.int = TRUE, xlab="Days", break.x.by = 30, linetype = "strata", surv.median.line = "hv", ggtheme = theme_classic2(base_family ="MaruBuri-Bold"), font.family = "MaruBuri-Bold", palette = c("#FC0000", "#6E888E"), data=sol)