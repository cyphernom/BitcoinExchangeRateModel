*(c)2020-2022 btconometrics*
import delimited https://coinmetrics.io/newdata/btc.csv, clear

*fix date formats and tsset the data*
gen d = date(time, "YMD")
drop time
rename d date
format %td date
tsset date

*we will extend the range a bit
tsappend, add(100000)



replace blkcnt = 6*24 if blkcnt == .

gen double sum_blocks = sum(blkcnt)
gen double hp_blocks = mod(sum_blocks, 210001)

gen hindicator = 0
replace hindicator = 1 if hp_blocks <200 & hp_blocks[_n-1]>209000

gen epoch =sum(hindicator) -1

gen double reward = 50/(2^epoch)

replace reward = 0 if epoch >= 33

gen double daily_reward = blkcnt * reward
gen double tsupply = sum(daily_reward)
sort date

replace double logprice = ln(priceusd)
*first, lets assume rewards are less important over time (due to there being less and less)
*but also, lets assume that there is exponential growth on each epoch. 
*the reason for this assumption is fundamentals 
gen double phasevariable = reward-(epoch+1)^2

*on the basis it takes abotu 30 years for stuff to play out (i.e. 10957 days)
gen double phaseplus = phasevariable +((date-d(31oct2008)) /10957)^4

*thjs results in the model never reaching infinity, and slowing down growth substantially after some time
drop if date<d(18jul2010)
reg logprice l(1).logprice phaseplus if epoch<2

  gen double decayfunc = 3*exp(-0.0004*[_n])*cos(0.005*[_n]-1)
  gen double secondorderdecayfunc = 0
  *1.5*exp(0.9-0.0005*[_n])*cos(-0.0055*[_n]-1)
 
mat coefs = e(b)

gen double YHAT = logprice
replace YHAT = coefs[1,3]+coefs[1,2]*phaseplus+coefs[1,1]*YHAT[_n-1] if date >d( 30jul2010 )
replace YHAT = YHAT+decayfunc+secondorderdecayfunc
gen  double eYHAT = exp(YHAT)
tsline priceusd eYHAT  if date>d(1jan2021) & date<d(1jan2023)
