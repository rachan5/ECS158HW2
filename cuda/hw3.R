
x <- c(1, 1,2,2, 3,3, 4,4, 5,5, 6,6, 7,7, 8,8, 9,9, 10,10)
y <- c(11,11, 12,12, 13,13, 14,14, 15,15, 16,16, 17,17, 18,18, 19,19, 20,20)

smoother <- function(x,y,h) {
   meanclose <- function(t) 
      mean(y[abs(x-t) < h])
   sapply(x,meanclose)
}

print(smoother(x,y,2))
