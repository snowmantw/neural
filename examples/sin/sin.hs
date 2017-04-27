{-# LANGUAGE DataKinds #-}

import           Control.Category
import           Control.Monad.Random
import           Data.MyPrelude
import           Data.Utils
import           Graph
import           Numeric.Neural
import qualified System.Console.ANSI  as ANSI
import           Prelude              hiding ((.))

main :: IO ()
main = flip evalRandT (mkStdGen 739570) $ do
    let xs = [0, 0.01 .. 2 * pi]
    m <- modelR $ whiten sinModel xs  -- modelR: Generates a model with randomly initialized weights
    runEffect $
            simpleBatchP [(x, sin x) | x <- xs] 10  -- "Producer". Samples available, then the "mini-batch size" (amount per batched procedure I think)
        >-> descentP m 1 (const 0.5)                -- the model, then "first-generation", learning rate *function*; while P denotes "Pipe"
        >-> reportTSP 50 report                     -- TS denotes "tranining state". Given report interval and report action
        >-> consumeTSP check                        -- "Consume". The training state to decide whether it's ended, and return a value.
                                                    -- Given the checking function for model to see if it's finished

  where
    
    -- With data constructor that receives specific kind of nerual layer (vector), the differatable functions for it, 
    -- (b -> f Double) ? for pure
    -- (g Double -> c) ? for vhead
    --
    -- while `g` now is from "Component f g" can be decomposed from the (tanhLayer . (tanhLayer :: Layer 1 4)) 
    -- "Component f g": behave like 'Arrow's with choice over a different base category --> That is, `f` and `g` denote
    -- the parameters of a parameterized differentiable function that `f Double -> g Double` (?).
    --
    -- More explainations: F(x) = f(g(x)), while F is differtiable and can apply chain-rule to get GD done.
    -- 
    -- (why two functions `f` and `g` is because the non-linear composition as the essence of NN)
    -- (and once it can handle two, one of them can be subsituted with more like g(x) = h(w(y....(x))))
    --
    -- ParamFun: `s`, `t` types for parameters while `a` and `b` is for I/O values.
    --
    -- (with `weights`, `compute`, and `initR`)
    --
    -- Layer: from Vector length 1 to length 4; then combile into two tanhLayers for this model.
    -- -> It's by extension DataKinds:
    -- -> which means the :: Vector 4 2 create a type restriction on type-level. Without the extension,
    -- -> all kinds like `* -> *` is actually not useful.
    -- -> ("This means we can have untyped functional programming at the type level.")
    -- -> http://stackoverflow.com/questions/20558648/what-is-the-datakinds-extension-of-haskell
    --
    -- Note that composition and (>>>) is from category, to make `cat a b -> cat b c -> cat a c`.
    --
    -- So the signature of StdModel is inputs/outputs and their types.
    --
    --
    -- From the Input/Hidden/Output model, we only specify the input and output in the signature.
    -- We specify another one in the first hidden layer. So unless the latest hidden layer
    -- automatically generated from the output and the previous one, or there must be something unknown
    -- about how it decides the dimension. 
    --
    sinModel :: StdModel (Vector 1) (Vector 1) Double Double
    sinModel = mkStdModel
        (tanhLayer . (tanhLayer :: Layer 1 4))
        (\x -> Diff $ Identity . sqDiff (pure $ fromDouble x))    -- DFunc for tanhLayer. Input is the output computed from samples.
        pure                                                      -- How to extract (ex: Double here) from samples for input layer.
        vhead                                                     -- How to extract result from output layer.

    -- The "error" function
    -- Receive training state and extract the "updated model"; get the computed output from "modelled function (our sine)"
    -- The maximum is to pick the most wrong value from the whole inputs
    getError ts =
        let m = tsModel ts
        in  maximum [abs (sin x - model m x) | x <- [0, 0.1 .. 2 * pi]]

    -- A pure IO function that output the current result on the screen.
    -- Can be embedded into our training pipleline because the whole process is piped.
    report ts = liftIO $ do
        ANSI.clearScreen
        ANSI.setSGR [ANSI.SetColor ANSI.Foreground ANSI.Vivid ANSI.Red]
        ANSI.setCursorPosition 0 0
        printf "Generation %d\n" (tsGeneration ts)
        ANSI.setSGR [ANSI.Reset]
        graph (model (tsModel ts)) 0 (2 * pi) 20 50

    -- If it converges to a function can produce result with only 0.1 error diff,
    -- we say it's successfully trained.
    check ts = return $ if getError ts < 0.1 then Just () else Nothing
