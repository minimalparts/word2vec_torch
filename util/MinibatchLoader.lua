-- Modifed from Karpathy's char-rnn, itself...
-- Modified from https://github.com/oxford-cs-ml-2015/practical6
-- the modification included support for train/val/test splits

function numkeys(T)
    local count = 0
    for _ in pairs(T) do count = count + 1 end
    return count
end

local MinibatchLoader = {}					-- AH: Define namespace, so functions can be called e.g. MinibatchLoader.nextbatch(). Namespaces in lua are tables.
MinibatchLoader.__index = MinibatchLoader			-- AH: __index is a metamethod to provide a result even when lua doesn't find a provided key in its table. 
									--     In this case, calling a MinibatchLoader that doesn't exist will simply return a MinibatchLoader object.

function MinibatchLoader.create(data, vocab, batch_size, seq_length)
    -- split_fractions is e.g. {0.9, 0.05, 0.05}

    local self = {}							-- AH: MinibatchLoader is a table
    setmetatable(self, MinibatchLoader)					-- AH: Lua's tables only have a limited set of operations defined over them. Creating a metatable allows us to define extra functions
									--     on a table. Whenever we call a function not defined for tables, Lua looks for an appropriate metatable.

    -- cut off the end so that it divides evenly
    local len = numkeys(data)						
    print("Length of data",len)
    if len % (batch_size * seq_length) ~= 0 then			
        print('cutting off end of data so that the batches/sequences divide evenly')
        data = data:sub(1, batch_size * seq_length 
                    * math.floor(len / (batch_size * seq_length)))
    end

    -- count vocab
    self.vocab_size = 0
    for _ in pairs(self.vocab_mapping) do 				-- AH: iterate through all pairs in vocab_size and count them
        self.vocab_size = self.vocab_size + 1 
    end

    -- self.batches is a table of tensors
    print('reshaping tensor...')
    self.batch_size = batch_size
    self.seq_length = seq_length

    local ydata = data:clone()						-- AH: data='This is a test', ydata='This is a test'
    ydata:sub(1,-2):copy(data:sub(2,-1))				-- AH: ydata:sub(1,-2) => 'This is a tes', data:sub(2,-1) => 'his a test'
    ydata[-1] = data[1]							-- AH: ydata[-1] => 't', data[1] => 'T'. So now, ydata => 'his a testT' ???
									--     if one argument is -1, its value is inferred from the value of other argument(s) 
    self.x_batches = data:view(batch_size, -1):split(seq_length, 2)  -- #rows = #batches
									-- AH: 'view' creates a view with different dimensions of the storage associated with tensor
									--     if one argument is -1, its value is inferred from the value of other argument(s)
									--     So get a matrix where each row is a batch and then split it in chunks of n columns where n is seq_length
									--     split() is a funny function. For a 3x4 matrix x, x:split(1,2) splits dimension 2 (4 columns) into bits of size 1,

    -- print(batch_size)
    -- print(#self.x_batches)									--     so we get 4 tensors of size 3x1.
    self.nbatches = #self.x_batches
    self.y_batches = ydata:view(batch_size, -1):split(seq_length, 2)  -- #rows = #batches
    assert(#self.x_batches == #self.y_batches)				-- AH: error handling.'assert' returns error if condition not satisfied

    -- lets try to be helpful here
    if self.nbatches < 50 then
        print('WARNING: less than 50 batches in the data in total? Looks like very small dataset. You probably want to use smaller batch_size and/or seq_length.')
    end


    collectgarbage()
    return self
end

function MinibatchLoader:reset_batch_pointer(split_index, batch_index)
    batch_index = batch_index or 0							-- AH: batch_index as provided, or 0
    self.batch_ix[split_index] = batch_index						-- AH: batch_ix for train/val/test split set to batch_index
end

function MinibatchLoader:next_batch()
    self.nbatches = self.batch_ix[split_index] + 1				-- AH: increment batch id
    if self.batch_ix[split_index] > self.split_sizes[split_index] then
        self.batch_ix[split_index] = 1 -- cycle around to beginning
    end
    -- pull out the correct next batch
    local ix = self.batch_ix[split_index]
    if split_index == 2 then ix = ix + self.ntrain end -- offset by train set size
    if split_index == 3 then ix = ix + self.ntrain + self.nval end -- offset by train + val
    return self.x_batches[ix], self.y_batches[ix]					-- AH: return batch x, and batch y (shifted by one character)
end


return MinibatchLoader

