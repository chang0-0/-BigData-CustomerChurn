#!/usr/bin/env python3
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf


# 로지스틱 회귀 분석 모델
# churn 부분 True와 False로 구성되어 있음
# 새로운 churn01 컬럼 부분 생성 0.과 1.의 부분으로 구성되어 
# 1. == True z > 0 (양성 || 고객 이탈)
# 0. == False z < 0 (음성 || 고객 잔류)
# z == 0 (로지스틱 중간값 0.5 해당)

# Read the data set into a pandas DataFrame
churn = pd.read_csv('churn.csv', sep=',', header=0)
churn.columns = [heading.lower() for heading in churn.columns.str.replace(' ', '_').str.replace("\'", "").str.strip('?')]

churn['churn01'] = np.where(churn['churn'] == 'True.', 1., 0.) 
#chrun으로 되어있으면 아니면 0으로 해라 로지스틱 회귀분석은 숫자.  0또는 1로 바꿔야됨 False True는 계산이안됨 False 는 0이됨.
print(churn.head())
print(churn.describe())
print(churn.info())


# Calculate descriptive statistics for grouped data
print(churn.groupby(['churn'])[['day_charge', 'eve_charge', 'night_charge', 'intl_charge', 'account_length', 'custserv_calls']].agg(['count', 'mean', 'std']))
#aggligation



# 변수별로 서로 다른 통계량 구하기
# Specify different statistics for different variables
print(churn.groupby(['churn']).agg({'day_charge' : ['mean', 'std'], 
				'eve_charge' : ['mean', 'std'],
				'night_charge' : ['mean', 'std'],
				'intl_charge' : ['mean', 'std'],
				'account_length' : ['count', 'min', 'max'],
				'custserv_calls' : ['count', 'min', 'max']}))

# Create total_charges, split it into 5 groups, and
# calculate statistics for each of the groups

churn['total_charges'] = churn['day_charge'] + churn['eve_charge'] + \
						 churn['night_charge'] + churn['intl_charge']

#결과 값에 식을 만들어봐라

'''
factor_cut = pd.cut(churn.total_charges, 5, precision=2)
def get_stats(group):
	return {'min' : group.min(), 'max' : group.max(),
			'count' : group.count(), 'mean' : group.mean(),
			'std' : group.std()}
grouped = churn.custserv_calls.groupby(factor_cut)
print(grouped.apply(get_stats).unstack())

# Split account_length into quantiles and
# calculate statistics for each of the quantiles
factor_qcut = pd.qcut(churn.account_length, [0., 0.25, 0.5, 0.75, 1.])
grouped = churn.custserv_calls.groupby(factor_qcut)
print(grouped.apply(get_stats).unstack())

# Create binary/dummy indicator variables for intl_plan and vmail_plan
# and join them with the churn column in a new DataFrame
intl_dummies = pd.get_dummies(churn['intl_plan'], prefix='intl_plan')
vmail_dummies = pd.get_dummies(churn['vmail_plan'], prefix='vmail_plan')
churn_with_dummies = churn[['churn']].join([intl_dummies, vmail_dummies])
print(churn_with_dummies.head())

# Split total_charges into quartiles, create binary indicator variables
# for each of the quartiles, and add them to the churn DataFrame
qcut_names = ['1st_quartile', '2nd_quartile', '3rd_quartile', '4th_quartile']
total_charges_quartiles = pd.qcut(churn.total_charges, 4, labels=qcut_names)
dummies = pd.get_dummies(total_charges_quartiles, prefix='total_charges')
churn_with_dummies = churn.join(dummies)
print(churn_with_dummies.head())

# Create pivot tables
print(churn.pivot_table(['total_charges'], index=['churn', 'custserv_calls']))
print(churn.pivot_table(['total_charges'], index=['churn'], columns=['custserv_calls']))
print(churn.pivot_table(['total_charges'], index=['custserv_calls'], columns=['churn'], \
						aggfunc='mean', fill_value='NaN', margins=True))
'''
# Fit a logistic regression model

my_formula = 'churn01 ~ account_length + custserv_calls + total_charges'
from statsmodels.formula.api import ols, glm, logit


print("========================================================== 표준화 되기 전 ==========================================================")
# 원래 기존의 코드 (표준화 안된 코드)
# Fit a logistic regression model

# 로지스틱 하기전에 라이브러리를 통해서 전체 데이터 표준화 작업 진행



dependent_variable = churn['churn01']
independent_variables = churn[['account_length', 'custserv_calls', 'total_charges']]
independent_variables_with_constant = sm.add_constant(independent_variables, prepend=True)
print(independent_variables_with_constant)
logit_model = sm.Logit(dependent_variable, independent_variables_with_constant).fit()
print(logit_model.summary())
# print("\nQuantities you can extract from the result:\n%s" % dir(logit_model))
print("\nCoefficients:\n%s" % logit_model.params)
print("\nCoefficient Std Errors:\n%s" % logit_model.bse)

# 결과 값 테스트 중 상수항을 넣지 않고 테스트 결과
# 상수항을 넣으면 회귀분석에서 좋다고 하는데, 결과가 변하지 않고 그대로 출력해도 상관없는듯

print("========================================================== 라이브러리 사용 테스트 ========================================================== ")
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
# 라이브러리 사용할거면 여기서 부터 진행하면 됨.
scaler.fit(independent_variables_with_constant)

independent_variables_with_constant = scaler.transform(independent_variables_with_constant)

independent_variables_standardized = scaler.transform(independent_variables_with_constant)

print(independent_variables_standardized)

logit_model = sm.Logit(dependent_variable, independent_variables_standardized).fit()
print(logit_model.summary())


#합쳐서 변수에 저장함. 종속변수와 표준화 된 값. axis=1으로 하나로 합해서 합침


#여기 까지가 표준화 된 값 출력.
logit_model = logit(independent_variables_standardized, data=churn).fit()
print(logit_model)





#logit_model = smf.glm(output_variable, input_variables, family=sm.families.Binomial()).fit()
#logit_marginal_effects = logit_model.get_margeff(method='dydx', at='overall')
#print(logit_marginal_effects.summary())


print("========================================================== 표준화 전 전처리 작업 ==========================================================")
lm = logit(my_formula, data=churn).fit()
print(lm.summary())

# 계수 출력 함수
print("\nCoefficients:\n%s" % lm.params)
# 계수 오류 출력
print("\nCoefficient Std Errors:\n%s" % lm.bse)

print("========================================================== 표준화 작업 ==========================================================")

# 표준화 작업 시작. 
# 종속 변수 지정 dependent_var = churn01 열 지정
dependent_variable = churn['churn01']
# independent_var = churn01을 제외한 나머지 열 모두 지정.
independent_variables = churn[churn.columns.difference(['churn01'])]






# 표준화 공식 적용 (종속변수 - 평균값) / 표준 편차 = 표준화 z-score.
independent_variables_standardized = (independent_variables - independent_variables.mean()) / independent_variables.std()

#합쳐서 변수에 저장함. 종속변수와 표준화 된 값. axis=1으로 하나로 합해서 합침
customer_standardized = pd.concat([dependent_variable, independent_variables_standardized], axis=1)

#여기 까지가 표준화 된 값 출력.
logit_model = logit(my_formula, data=customer_standardized).fit()





print(customer_standardized.describe())
print(logit_model.summary())

# 계수 출력 함수
print("\nCoefficients:\n%s" % logit_model.params)
# 계수 오류 출력
print("\nCoefficient Std Errors:\n%s" % lm.bse)


print ("========================================================== 라이브러리를 사용한 표준화 작업 ==========================================================")



'''
print("\nQuantities you can extract from the result:\n%s" % dir(logit_model))
print("\nCoefficients:\n%s" % logit_model.params)
print("\nCoefficient Std Errors:\n%s" % logit_model.bse)
#logit_marginal_effects = logit_model.get_margeff(method='dydx', at='overall')
#print(logit_marginal_effects.summary())
'''
print("======================================= 로지스틱 함수 식 =======================================")
print("\ninvlogit(-7.2205 + 0.0012*mean(account_length) + 0.4443*mean(custserv_calls) + 0.0729*mean(total_charges))")


# Fit Standardized 
def inverse_logit(model_formula):
	from math import exp
	return (1.0 / (1.0 + exp(-model_formula))) * 100

at_means = float(logit_model.params[0]) + \
	float(logit_model.params[1])*float(churn['account_length'].mean()) + \
	float(logit_model.params[2])*float(churn['custserv_calls'].mean()) + \
	float(logit_model.params[3])*float(churn['total_charges'].mean())

print("")
print("======================================= 평균값 계산 =======================================")
print(churn['account_length'].mean())
print(churn['custserv_calls'].mean())
print(churn['total_charges'].mean())
print(at_means)
print("Probability of churn when independent variables are at their mean values: %.2f" % inverse_logit(at_means))

cust_serv_mean = float(logit_model.params[0]) + \
	float(logit_model.params[1])*float(churn['account_length'].mean()) + \
	float(logit_model.params[2])*float(churn['custserv_calls'].mean()) + \
	float(logit_model.params[3])*float(churn['total_charges'].mean())
		
cust_serv_mean_minus_one = float(logit_model.params[0]) + \
		float(logit_model.params[1])*float(churn['account_length'].mean()) + \
		float(logit_model.params[2])*float(churn['custserv_calls'].mean()-1.0) + \
		float(logit_model.params[3])*float(churn['total_charges'].mean())

print(cust_serv_mean)
print(churn['custserv_calls'].mean()-1.0)
print(cust_serv_mean_minus_one)
print("Probability of churn when account length changes by 1: %.2f" % (inverse_logit(cust_serv_mean) - inverse_logit(cust_serv_mean_minus_one)))


# Predict churn for "new" observations
print("======================================= 값 예측하기 =============================================")
new_observations = churn.loc[churn.index.isin(range(10)), independent_variables.columns]
new_observations_with_constant = sm.add_constant(new_observations, prepend=True)
y_predicted = logit_model.predict(new_observations_with_constant)
y_predicted_rounded = [round(score, 2) for score in y_predicted]
print(y_predicted_rounded)


# Fit a logistic regression mode
''''이거 표준화 아니라고 함.'''
output_variable = churn['churn01']
vars_to_keep = churn[['account_length', 'custserv_calls', 'total_charges']]
inputs_standardized = (vars_to_keep - vars_to_keep.mean()) / vars_to_keep.std()
input_variables = sm.add_constant(inputs_standardized, prepend=False)
logit_model = sm.Logit(output_variable, input_variables).fit()

#표준화 진행 해야됨. 
from sklearn.preprocessing import StandardScaler
'''
표준화 도와주는 라이브러리
1순위 어떤 값이 종속변수의 값에 영향을 많이 주는 지 파악하기
2순위 그 중에서 표준편차가 큰 값 파악하기
3순위 

'''

# logit_model = smf.glm(output_variable, input_variables, family=sm.families.Binomial()).fit()
print(logit_model.summary())
print(logit_model.params)
print(logit_model.bse)
logit_marginal_effects = logit_model.get_margeff(method='dydx', at='overall')
print(logit_marginal_effects.summary())

# Predict output value for a new observation based on its mean standardized input values
input_variables = [0., 0., 0., 1.]
predicted_value = logit_model.predict(input_variables)
print("Predicted value: %f", predicted_value) 
