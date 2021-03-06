--[[
Config file for skipgram with negative sampling 
--]]

require("io")
require("os")
require("paths")
require("torch")
dofile("w2v.nonce.lua")

-- Default configuration
config = {}
config.corpus = "corpus.txt" -- input data
config.background = "none" -- background space
config.window = 2 -- (maximum) window size
config.dim = 100 -- dimensionality of word embeddings
config.alpha = 0.75 -- smooth out unigram frequencies
config.table_size = 1e8 -- table size from which to sample neg samples
config.neg_samples = 5 -- number of negative samples for each positive sample
config.lr = 0.5 -- initial learning rate
config.min_lr = 0.001 -- min learning rate
config.epochs = 1 -- number of epochs to train
config.gpu = 0 -- 1 = use gpu, 0 = use cpu
config.stream = 1 -- 1 = stream from hard drive 0 = copy to memory first
config.batch_size = 1
config.subsampl = 10^-4
config.probe = "UNK"
config.save = "./models/chimera.model.th"

-- Parse input arguments
cmd = torch.CmdLine()
cmd:option("-corpus", config.corpus)
cmd:option("-background", config.background)
cmd:option("-window", config.window)
cmd:option("-dim", config.dim)
cmd:option("-lr", config.lr)
cmd:option("-min_lr", config.min_lr)
cmd:option("-neg_samples", config.neg_samples)
cmd:option("-table_size", config.table_size)
cmd:option("-epochs", config.epochs)
cmd:option("-gpu", config.gpu)
cmd:option("-stream", config.stream)
cmd:option("-batch_size", config.batch_size)
cmd:option("-subsampl", config.subsampl)
cmd:option("-alpha", config.alpha)
cmd:option("-probe",config.probe)
cmd:option("-save",config.save)
params = cmd:parse(arg)

for param, value in pairs(params) do
    config[param] = value
end

--for i,j in pairs(config) do
--    print(i..": "..j)
--end
-- Train model
m = Word2Vec(config)
--m = torch.load(config.save)
--m:reinit(config)
m:build_vocab(config.corpus,config.background)
m:build_table_current()
--m:build_table_background()

--m:print_sim_words({"___"},10)
for k = 1, config.epochs do
    m.lr = config.lr -- reset learning rate at each epoch
    m:train_model(config.corpus)
    m:print_sim_words({"___"},10)
end
print("Sim with "..config.probe..": "..m:cosine("___",config.probe))
print("Rank "..m:get_rank_sim("___",config.probe))
--torch.save(config.save,m)
--m:print_semantic_space()
