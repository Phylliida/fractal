# Messages from the user

Notes Danielle sent while this loop was running. Most recent at the bottom.

## 2026-06-22T20:34:53Z

inspect md document to see if any of the tips there can be used, and migrate code to using a glsl shader (if that speeds it up, especially for deep zooms)

## 2026-06-22T21:23:49Z

can u add input field for iteration count and make it so any zoom immediately cancels the running task and doesn't run any more until zoom is done, zoom should just render scaled current render

## 2026-06-22T21:53:00Z

point filter (not bilinear) fractal so more crisp (especially when zooming), add supersampling, optimize fractal gpu rendering around like 2^270, see eg http://100.68.218.82:8111/#re=-1.369078017863660784890619576747781310848768032841633323730495873496232879296538490243106365484246242476783355722&im=-0.071817675972918479944583194368632476442138106251769795140812120871593742404751576456750164324645880810732436640&r=5.99473960e-82&i=28500&p=ultra&cy=48&sh=0 u should have access to gpu now

## 2026-06-22T22:53:59Z

add click-to-zoom in clicked location and optimize the deep zoom pertubation gpu shaders until it can run fast on swiftshader

## 2026-06-23T00:23:31Z

let's optimize the gpu deep zoom further

## 2026-06-23T02:03:52Z

let's let us zoom in past 2^218

## 2026-06-23T03:17:59Z

could you figure out why gpu not working (you have access to one) for the tests so you can profile the shaders

## 2026-06-23T23:16:42Z

let's properly fix the gpu bug

## 2026-06-24T02:47:28Z

optimize the gpu code when deep zoom (2^500)

## 2026-06-30T04:39:24Z

optimize the shader further, and don't have it render black screen instead replace low res with high res as it comes in

## 2026-06-30T05:27:37Z

generate new ideas of how it can be optimized further and benchmark them, repeat until we get 900% faster

## 2026-07-02 (live interrupt during Spawn 27)

I think what I'm most interested in is optimizing the shader further
