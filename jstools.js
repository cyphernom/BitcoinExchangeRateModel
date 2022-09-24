/*NB REQUIRES JSTAT AND PLOTLY LIBS!*/

function main(days, isf, divid, title) {

    const sd = (arr, usePopulation = false) => {
      const mean = arr.reduce((acc, val) => acc + val, 0) / arr.length;
      return Math.sqrt(
        arr.reduce((acc, val) => acc.concat((val - mean) ** 2), []).reduce((acc, val) => acc + val, 0) /
          (arr.length - (usePopulation ? 0 : 1))
      );
    };

    function deepClone(X) {
      return JSON.parse(JSON.stringify(X));
    }

    /*
     * pre:
     *  x is a vector
     *  centre is a pivot point
     *
     * post:
     *  returns an array filled with the likelihood of each point in x, given the history of x
     */
    function getLikelihood(x, centre) {
      var c = centre|0;
      var output =[];
      for(var i = 0; i< x.length; ++i) {
        var ov = x.slice(0,i).filter(counted=>counted>=x[i]).length;
        var uv = x.slice(0,i).filter(counted=>counted<=x[i]).length;
        if(x[i]<c) {
            output.push(getNPLikelihood(uv+1,i+1,0.05)); //+1 shift from zero for true df
          } else if(x[i]>=c) {
            output.push(getNPLikelihood(ov+1,i+1,0.05));
          }
        }
        return output;
    }

    function egranger(y, x) {

     // var molslm = OLS(y,x);
      var olslm = OLS(y,x);
      var residlm = olslm.residuals;
      var Dresidlm = diff(residlm,1);
      var Lresidlm = lag(residlm,1);//.slice(1,Dresid.length+1);
      var n = Dresidlm.length;
      var egcmlm = OLS(Dresidlm,Lresidlm, false); //no intercept
      var taus=egcmlm.beta1_tstat;
      return taus;
    }



    function deepClone(X) {
      return JSON.parse(JSON.stringify(X));
    }
    function lag(_x,by) {
      var x = deepClone(_x);
      var d = new Array(x.length-by);
      for(i = by;i<x.length-by;i++){
        d[i] =Number(x[i-by].toPrecision(4));
      }
      return d;
    }

    function diff(_x, by) {
      var x = deepClone(_x);
      var d = new Array(x.length-by);
      for(i = by;i<x.length-by;i++){
        d[i] =Number((x[i]-x[i-by]).toPrecision(4));
      }
      return d;
    }

    function minimum(value) {
          if (toString.call(value) !== "[object Array]")
          return false;
          return Math.min.apply(null, value);
       }

    function maximum(value) {
          if (toString.call(value) !== "[object Array]")
          return false;
          return Math.max.apply(null, value);
       }
      //implements a simple percentile function
      //x is a vector, p is the probability
      function percentile(_x, p) {
        //because we dont want to be mutating x
        var x = deepClone(_x);
    //take a shortcut if p >= 1 or p<= 0
      if(p>=1) {
          return (maximum(x));
        } if (p<=0) {
          return (minimum(x));
          }else {
        //first get the rank
        var n = x.length;
        var r = (1+ p*(n-1))-1; //-1 for zero addressing

        //now split the rank into integer and fractional components
        function f(n){ return Number("0."+String(n).split('.')[1] || 0); }
        function i(n){ return Number(String(n).split('.')[0] || 0); }

        var rfrac = f(r);
        var rint = i(r);

        //now sort the vector ascending
        x.sort(function(a, b){return a-b});

        //now interpolate for the correct position:
        var xp = (1-rfrac)*x[rint]+rfrac*x[rint+1];
        return xp;
      }
      }

    //stepping through the set of reals to find the solution
    //probably could go to nlogn if i did binary search but meh
    function binominv(n,p,a) {
      var setOfReals=[];
      //big enough for most purposes!
      for (j=0;j<1000000000000;j++) {
       var result = jStat.binomial.cdf(j,n,p);
          if(result>a) {
            console.log(j);
            return(j);
          }
        }
    }

    // gives a 1-alpha upper CI on the pth quantile of X
    /*
     * Lower quantile confidence interval
    =percentile(X,(binom.inv (n,p,1-alpha/2)+1)/n

    Upper quantile confidence interval
    =percentile(X,( binom.inv (n,p,alpha/2)+1)/n


    Where
    ?	X is vector
    ?	n is length X
    ?	p is the quantile of interest
    ?	alpha is 1-coverage
    ?	binom.inv parameters = (trials, probability_success, alpha)
    */
    function upperCI(alpha, p, _x) {
      var X = deepClone(_x);
      var n = X.length;
      return percentile(X,binominv(n,p,1-alpha/2)/n);
    }

    //taking the complement for the lower bound
    function lowerCI(alpha, p, _x) {
      var X = deepClone(_x);
      var n = X.length;
      return percentile(X,binominv(n,p,alpha/2)/n);
    }


      function mean(_x) {
       var X = deepClone(_x);
      return (X.reduce((previous, current) => current += previous)/X.length);
    }


      function lagf(_x,by) {
      var x = deepClone(_x);
      var d = new Array(x.length-by);
      for(i = 0;i<x.length-by;i++){
        d[i] =x[i-1+by];
      }
      return d;
    }

    //returns the (non-parametric) likelihood that there are x nonconforming units in n
        function getNPLikelihood(x,n, alpha) {
          return jStat.beta.inv( alpha, x, n-x+1 );
        }
        /*old diff
    function diff(_x, by) {
      var d= [];
      for(i = by;i<_x.length;i++) {
        d[i-by] = _x[i]-_x[i-by]
      }
      return d;
    }*/
    function wrarr(_X) {
      var X = deepClone(_X);
      var o = [];
      for(var i = 0; i<X.length;i++) {
        o[i] = [X[i]];
      }
      return(o);
    }

    function mOLS(_Y, _X) {
       var Y = deepClone(_Y);
       var X = deepClone(_X)
      //Multivariable linear regression
      //using linear algebra
      //first we need to add a constant to _X
      //if X is 2d matrix array [[x1,x2]...n]
      //then we need to insert another column of 1s
      // i.e. [[1, x1, x2]...n]
      var n = _X.length
      var constant = Array(n).fill().map((_, i) => 1);
     // console.log(constant)
      var x = [];
      var ncol = _X[0].length;
     // console.log(ncol)
      x.push(constant)
       for(var i = 0; i<ncol;i++) {
         x.push(X.map(function(value,index) { return value[i]; }));
        // console.log("adding col");
       }
       var xt = x;
       x = math.transpose(x)
       var xtx = math.multiply(xt,x);
       var xtxinv = math.inv(xtx);
       var xtxinvxt = math.multiply(xtxinv, xt);
       var betahat = math.multiply(xtxinvxt,_Y);
    // console.log(xtxinvxt);
       //also nice to get the hat matrix
       // (i.e. : X(XTX)âˆ’1XT)
       var hatmatrix = math.multiply(x,xtxinv)
       hatmatrix = math.multiply(hatmatrix, xt);
       var yhat = math.multiply(hatmatrix, _Y);
       //leverage is the trace of the hat matrix
       var leverage = [];
       for(var h = 0; h< hatmatrix.length; h++) {
         leverage.push(hatmatrix[h][h]); //hat matrix will be square
       }
       var residuals = [];
       var SSE = 0;
       var SST = 0;
       var ybar = mean(_Y.map(function(value,index) { return value[0]; }));
       for(var i = 0; i<_Y.length;i++) {
           residuals.push(_Y[i]-yhat[i]);
           SSE+=Math.pow(residuals[i],2);
           SST+=Math.pow(ybar-Y[i],2);
       }
       var R2 = 1-SSE/SST;
             AIC = n*Math.log(SSE/n)+ (n + ncol) / (1 - (ncol + 2) / n)//Corrected AIC (Hurvich and Tsai, 1989)
             SSR = SST-SSE;
      var lm = {}
       lm['betahat']=betahat;
       lm['yhat'] = yhat;
       lm['fitted']=yhat;//compatability
       lm['hat_matrix'] = hatmatrix;
       lm['leverage'] = leverage;
       lm['residuals'] = residuals;
       lm['SSE'] = SSE;
       lm['SSR'] = SSR;
       lm['AIC'] = AIC;
       lm['x'] = x;
       lm['y']=_Y;
       lm['SST'] = SST;
       lm['R2'] = R2;

      return(lm);
    }
   function rnd(n, p) {
       return math.format(n,p);
   }
   //default value frm medium Articles
 
   //X0 = 0.0017
   //GAMMA = 0.016
   //test: X = X = [-0.45, 0.2];===-5.742113342065857
   //assume X is a price series unmodified vector then
   function ll(X_) {
   	var x = deepClone(X_);
   	var perc = (x-lag(x,1))/lag(x);
   	x0=0.0017;
   	gamma=0.016;
   	n = perc.length;
   	c = -n*Math.log(gamma*Math.PI);
   	k=0;
   	for(i = 0; i<n; i++) {
   		k+=Math.log(1+Math.pow(((perc[i]-x0)/gamma),2));
   	}
   	return c - k;
   }

   function ARDL(_Y, _X) {
    var Y = deepClone(_Y)
    var X= deepClone(_X)
     //basic ARDL 1,1 is
     //reg _Y _X L1._Y L2._Y
     //so need to add variables to _X assuming it is just s2f
     var y1lag = lagf(Y, 1); //rewrote lag function to be more useful for ardl
     y1lag = y1lag.slice(0,y1lag.length);

     var y2lag = lagf(y1lag, 1);
     y2lag = y2lag.slice(0,y2lag.length);

   var n = y2lag.length;

     var x = [];

     x.push(X.slice(2, X.length))
     x.push(y1lag.slice(1, y1lag.length));
     x.push(y2lag);

     x = math.transpose(x)
     m = mOLS(wrarr(Y.slice(2,Y.length)), x);
    // var lm={}
     //lm.model = m;
     //lm.residuals =
     var mu = m.betahat[0];
     var b1 = m.betahat[1];
     var g1 = m.betahat[2];
     var g2 = m.betahat[3];
     var lr = b1[0]/(1-g1[0]-g2[0]);
     var adj = (g1[0]-1+g2[0]);
     var sr = -g1[0];

     //can be used to construct the STATA EC form or the non EC form:
     var model_parameters = {}
     var deltayt = mu[0]+adj*(Y[Y.length-2]-lr*X[X.length-2])-g2[0]*(Y[Y.length-2]-Y[Y.length-3])+b1[0]*(X[X.length-1]-X[X.length-2])
     console.log("mu:"+mu);
     console.log("adj:"+adj);
     console.log("Yt:" + Y[Y.length-1]);
     console.log("Yt-1:"+Y[Y.length-2]);
     console.log("Yt-2:"+Y[Y.length-3]);
     console.log("Xt:" + X[X.length-1]);
     console.log("Xt-1:"+X[X.length-2]);
     console.log("Xt-2:"+X[X.length-3]);

     console.log("LR:" + lr);
     console.log("Xt-1:"+X[X.length-2]);
     console.log("g2:"+g2);
     console.log("deltaYt-1:"+(Y[Y.length-2]-Y[Y.length-3]));
     console.log("b1:"+b1);
     console.log("deltaXt"+(X[X.length-1]-X[X.length-2]));

     model_parameters['mu']=mu;
     model_parameters['b1'] = b1;
     model_parameters['g1']=g1;
     model_parameters['g2']=g2;
     model_parameters['lr']=lr;
     model_parameters['adj'] = adj;
     model_parameters['sr'] = sr;
     model_parameters['lm'] = m;
     model_parameters['dyt'] = deltayt
     return(model_parameters);
   }



   //simple OLS (Ordinary Least Squares) function
   //returns the OLS estimate for the single predictor case y~x
   //depreciated. use mOLS instead.
   function OLS(_Y,_X, intercept=true){
       var x = deepClone(_X);
         var y = deepClone(_Y);
     if(y.length!=x.length) throw  new Error("Y and X are of different lengths");
             var lm = {};
             var n = y.length;
             var xbar = mean(x);
             var ybar = mean(y);
             var beta_num = 0;
             var beta_den = 0;
             var beta1hat = 0;
             var beta0hat = 0;
             var yhat =new Array(y.length);
             var residuals = new Array(y.length);
             var SST = 0;
             var SSE = 0;
             var SSR = 0;
             var R2 = 0;
             var AIC = 0;
             var xx =new Array(y.length);
             var xy =new Array(y.length);
             if(!intercept) {
               //no intercept, then
               //beta1 is just mean(x*y)/mean(x*x)
               for(var i = 0; i <y.length; i++) {
                     xy[i] = x[i]*y[i];
                     xx[i] = x[i]*x[i];
                      beta_den += Math.pow((x[i]-xbar),2);
               }
               var meanx = mean(xx);
              // console.log("mean xx:"+meanx);
               var meanxy = mean(xy);
             //  console.log("meanxy:"+meanxy);

               beta1hat = meanxy/meanx;
              // console.log(beta1hat);

             } else {
               //multiple loops required
               //:-1st to get coefficients
               //:-2nd to get residuals and fitted
               for (var i = 0; i < y.length; i++) {
                 if(!Number.isNaN(x[i]) & !Number.isNaN(y[i])) { //skip nans
                   beta_num += (x[i]-xbar)*(y[i]-ybar);
                   beta_den += Math.pow((x[i]-xbar),2);
                 }
               }
               beta1hat = beta_num/beta_den;
               beta0hat = ybar - beta1hat * xbar;
             }

             for(var i =0;i<y.length;i++) {
               yhat[i]=beta0hat+beta1hat*x[i];
               residuals[i] = y[i]-yhat[i];
               SSE+=Math.pow(residuals[i],2);
               SST+=Math.pow(ybar-y[i],2);
             }

           var  sebeta1hat = Math.sqrt((SSE/(y.length-2))/beta_den);
           var  b1tstat = beta1hat/sebeta1hat

           var  sebeta0hat = Math.sqrt((SSE/(y.length-2))*(Math.pow(xbar,2)/beta_den))
           var  b0tstat = beta0hat/sebeta0hat

             AIC = n*Math.log(SSE/n)+4 //assuming k = 1 for simple case
             SSR = SST-SSE;
          //   console.log("SST:" +  SST + " SSE:"+SSE);
             R2 = 1- SSE/SST;
             lm['x'] = x;
             lm['y'] = y;
             lm['beta1hat'] = beta1hat;
             lm['beta0hat'] = beta0hat;
             lm['R2'] = R2;
             lm['SSE'] = SSE;
             lm['SST']=SST;
             lm['SSR']=SSR;
             lm['AIC']=AIC;
             lm['fitted']=yhat;
             lm['residuals']=residuals;
             lm['beta1_tstat']=b1tstat;
             lm['beta0_tstat']=b0tstat;
             //console.log(yhat);
           //  console.log("r2 "+R2);
             return lm;
   }
    function getRope(x,n, upper=true) {
        if(upper) {
        //  qbeta(α/2;x,n−x+1)
          return jStat.beta.inv( 0.025, x, n-x+1 );
        } else {
        //  qbeta(1−α/2;x+1,n−x)
          return jStat.beta.inv(1- 0.025, x+1, n-x );
        }
    }
//  console.log(" n=3820 x=1 rope:"+getRope(x=1, n=3820,upper=true))
//console.log("qbeta(0.02,1,5) 0.004032389 = " + jStat.beta.inv(0.02,1,5))
//console.log(" getRope(1,100,T)=0.000253146" + getRope(1,100,true))
    var xmlhttp = new XMLHttpRequest();
    xmlhttp.onreadystatechange = function() {
      var logdelta={y: [], x:[], name: 'log-delta', type: 'scatter',  mode: 'lines'};
      var rope={y: [], x:[], name: 'RoPE', type: 'scatter', yaxis: 'y2',  mode: 'lines'};
      var logprice={y: [], x:[], name: 'BTC/USD', type: 'scatter',  mode: 'lines'};
 var sim10k=[];
      if (this.readyState == 4 && this.status == 200) {
        var dateprice = JSON.parse(this.responseText);
        var dateprices = dateprice.prices;



         logprice.y = dateprices.map(function(value,index) { return Math.log(value[1]); })
        logdelta.y = diff(logprice.y ,1);
      var   prices= dateprices.map(function(value,index) { return (value[1]); })
    //  console.log(prices)
        var deltas =  diff(prices,1);

        var returns = [];
        for (var i = 1 ; i<prices.length-1;i++) {
          returns[i] = deltas[i]/prices[i-1];
        }
        returns[0]=0;
    //    console.log(returns)

         var dailyvolatility=sd(returns);
         var annualVolatility=dailyvolatility*Math.sqrt(365.25)
         var dailydrift=mean(returns);
         var annualDrift=dailydrift*365.25;

         for(var i = 0; i<10000;i++) {
           sim10k[i] = prices[prices.length-1]*Math.exp((annualDrift-0.5*Math.pow(annualVolatility,2))+annualVolatility* jStat.normal.sample(0,1));
        }
        logpred = sim10k.map(function(value) { return Math.log(value); })
      //  console.log(logpred);

        var mu = mean(logpred);
        var sigma = sd(logpred);
        f = (x, n) => +x.toPrecision(n)
        var fifthquant = f(jStat.lognormal.inv( 0.05, mu, sigma ),3);
        var nfifthquant = f(jStat.lognormal.inv( 0.95, mu, sigma ),3);
        var median = f(jStat.lognormal.inv( 0.5, mu, sigma ),3);

      var output = "We know under the GBM this distribution is log normal. Thus by taking the parameters for the mean and standard deviation, we can then estimate various moments of the distribution — notable, the 5th (at "+fifthquant+") and 95th (at "+nfifthquant+") quantiles and the median (at "+median+"). Thus we can be 90% confident that the Bitcoin price will fall between $"+fifthquant+" and $"+nfifthquant+", with a median expectation of around $"+median+" one year from now, should volatility and drift remain as the have been for the past 4 years."

      document.getElementById('gbm_info').innerHTML= output;

        logdelta.x = dateprices.map(function(value,index) {
          var d = new Date(value[0]);
          var datestring = + d.getFullYear() +"-" + ("0"+(d.getMonth()+1)).slice(-2) +"-"+ ("0" + d.getDate()).slice(-2)  + " " + ("0" + d.getHours()).slice(-2) + ":" + ("0" + d.getMinutes()).slice(-2)+ ":" + ("0" + d.getSeconds()).slice(-2)
          return datestring });
      //    console.log(logdelta.x)
          logprice.x=logdelta.x;

        for(var i = 0; i< logdelta.y.length+1; i++){
          var x = logdelta.y.slice(0,i).filter(counted => counted >= logdelta.y[i]).length;
        //  console.log("x:"+x+" i:"+i)
          if(logdelta.y[i]>0) {
            rope.y.push(getRope(x+1,i+1,true));
      //      console.log("rope:"+rope.y[i])
          } else {
           rope.y.push(getRope(x+1,i+1,false));
      //       console.log("rope:"+rope.y[i])
          }
        }
           logprice.y = dateprices.map(function(value,index) { return value[1]; })

          rope.x = dateprices.map(function(value,index) {
          var d = new Date(value[0]);
          var datestring = + d.getFullYear() +"-" + ("0"+(d.getMonth()+1)).slice(-2) +"-"+ ("0" + d.getDate()).slice(-2)  + " " + ("0" + d.getHours()).slice(-2) + ":" + ("0" + d.getMinutes()).slice(-2)+ ":" + ("0" + d.getSeconds()).slice(-2)
          return datestring });
      }


      if (xmlhttp.readyState==4) {
        if(!isf) {
        updateplot(logprice, rope, title);
          plotGBM(sim10k);
      }else {
        newplot(logprice, rope, title);
        plotGBM(sim10k);
      }
      }
    };



    xmlhttp.open("GET", "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart?vs_currency=usd&days="+days, true);
    xmlhttp.send();

     function newplot(logprice ,rope, ttl) {
        var data = [logprice, rope];
        var layout = {
           plot_bgcolor: "rgb(10,10,10)",
          paper_bgcolor: "rgba(0,0,0,0)",
          title: ttl,
          yaxis: {title: 'Price', type:'log', overlaying: 'y2'},
          yaxis2: {
            title: 'Prob(log-delta[n+1]>log-delta[n])',
            titlefont: {color: 'rgb(148, 103, 189)'},
            tickfont: {color: 'rgb(148, 103, 189)'},
            side: 'right'
          },
          xaxis: {
            type: 'date'
          }
        };
        Plotly.newPlot(divid, data, layout)
      }

      function updateplot(logprice ,rope, ttl) {
         var data = [logprice, rope];
         var layout = {
           plot_bgcolor: "rgb(10,10,10)",
           paper_bgcolor: "rgba(0,0,0,0)",
           title: ttl,
           yaxis: {title: 'Price',type:'log', overlaying: 'y2'},
           yaxis2: {
             title: 'Prob(log-delta[n+1]>log-delta[n])',
             titlefont: {color: 'rgb(148, 103, 189)'},
             tickfont: {color: 'rgb(148, 103, 189)'},
             side: 'right'
           },
           xaxis: {
             type: 'date'
           }
         };
         Plotly.newPlot(divid, data, layout)
       }

       function plotGBM(x) {
         var trace = {
             x: x,
             type: 'histogram',
             opacity: 0.5,
             histnorm: "probability density"
           };
           var layout = {
             title: "the probability of the price in 1 year",
             plot_bgcolor: "rgb(10,10,10)",
             paper_bgcolor: "rgba(0,0,0,0)",
             xaxis: {
               type: 'log',
               autorange: true
             }
           }
         var data = [trace];
         Plotly.newPlot('gbm_plot', data, layout);
       }


  }

