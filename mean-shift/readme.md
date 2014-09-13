Simple Mean Shift
=================

Shown here is an example in R using a simple gradient decent to find the
optimal mean that optimizes the cost function. Using the out of sample
data that would be the same as predicting that every value should have a
loss of 323.1402 which would result in validation cost value of 214.306.

Simillarly shown here is a simple gradient decent on the last 500
observations resulting in an cost in the low 200's. Clearly this marks a
baseline solution accuratley predicting coeficients of key contributing
factors should greatly reduce the score (assuming those factors aren't
dominated by noise!)

    # use only last 500 observations as a training set
    n=500
    y = as.matrix(y[(length(y)-n):length(y)])
    x = x[(length(y)-n):length(y),]

    # define the gradient using the penalty inside the optimization function
    grad <- function(x, y, b) {
      m <- nrow(y)
      penalty <- rep(1, m)
      penalty[ b - y < 0 ] = 10 # apply the penalty
      gradient <- (1/m) * sum(((b - y) * penalty))
      return(gradient[1])
    }

    # define gradient descent update algorithm
    grad.descent <- function(x, y, maxit, alpha, b){
      m <- nrow(y)
      
      # initialize
      totalCost = c()
      b_i = b
      
      for (i in 1:maxit) {
        
        # shift to the new b
        b_i <- b_i - alpha  * grad(x, y, b=b_i)
        
        # compute cost using the training data
        penalty <- rep(1, m)
        penalty[b_i - y < 0] = 10
        totalCost[i] <- (1/m) * sum(abs(y - b_i) * penalty)
      }
      return(list(totalCost,b_i))
    }



    results = grad.descent(x, y, maxit=600, alpha=0.01, b=100)
    b = results[[2]]
    b

    ## [1] 266

    # plot the fit
    plot( rep(b,length(y)), y,
         xlab = "predicted", ylab="actual")
    abline(0,1)

![plot of chunk
unnamed-chunk-2](./run-optimization_files/figure-markdown_strict/unnamed-chunk-21.png)

    plot(y, cex=0.2, main = "Yield Loss Predicted VS Actual", ylab="Y")
    points(rep(b,length(y)), col="red", cex=0.2)

![plot of chunk
unnamed-chunk-2](./run-optimization_files/figure-markdown_strict/unnamed-chunk-22.png)

    # plot the cost
    plot(results[[1]][1:length(results[[1]])], main="Cost Function", xlab="Iterations")

![plot of chunk
unnamed-chunk-2](./run-optimization_files/figure-markdown_strict/unnamed-chunk-23.png)

    final_training_score = results[[1]][length(results[[1]])]
    final_training_score

    ## [1] 140.5

Effectively this built a model that predicts every point to have the
same value.

Predict the validation set and compute cost
-------------------------------------------

    #### HOLDOUT VALIDATION SET ####

    # apply final alpha and plot

    # predict point and save
    y_val_predicted = rep(b,nrow(x_val))

    plot(y_val)
    abline(h=b, col="red")

![plot of chunk
unnamed-chunk-3](./run-optimization_files/figure-markdown_strict/unnamed-chunk-3.png)

    err = y_val_predicted - y_val

    #Custom residual sum
    err_underpredict = abs(err[err<0])*10;        #10x penalty
    err_overpredict = abs(err[err>=0]);

    overall_score = (sum(err_underpredict)+sum(err_overpredict))/nrow(err)

    # report the score
    overall_score

    ## [1] 230

This final overall score is on the validation data using the model
created on the test data above.
