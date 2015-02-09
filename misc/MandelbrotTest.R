in.mandelbrot.set <- function(c, iterations = 100, bound = 1000000)
{
  z <- 0
 
  for (i in 1:iterations)
  {
    z <- z ** 2 + c
    if (Mod(z) > bound)
    {
      return(FALSE)
    }
  }
 
  return(TRUE)
}


resolution <- 0.01
 
sequence <- seq(-1, 1, by = resolution)
 
m <- matrix(nrow = length(sequence), ncol = length(sequence))
 
for (x in sequence)
{
  for (y in sequence)
  {
    mandelbrot <- in.mandelbrot.set(complex(real = x, imaginary = y))
    m[round((x + resolution + 1) / resolution), round((y + resolution + 1) / resolution)] <- mandelbrot
  }
}
 
png('mandelbrot.png')
image(m)
dev.off()
