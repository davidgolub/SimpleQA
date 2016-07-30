
local functions = torch.class('dmn.io_functions')

-- decodes json
function functions.json_decode(html)
  assert(html ~= nil, "must specify html to decode")
  res = cjson.decode(html)
  return res
end

function functions.load_image(img_path, library_to_use)
 assert(img_path ~= nil, "Must specify image path to load from")
 local lib 

 -- get library to use
  if library_to_use == nil then 
    if python ~= nil then 
      lib = 'python'
    else  
      lib = 'image' 
    end 
  else 
    lib = library_to_use 
  end

 local new_img = nil 
 local img 
        --dmn.logger:print("Trying to load image from " .. img_path)
        -- Hacky error handling for image load issues
        functions.trycatch(
          function() 
            if lib == 'gm' then 
              img = gm.load(img_path, 'double')
            elseif lib == 'python' then
              img = dmn.image_functions.python_load_image(img_path)
            else
                 local ok, input = pcall(function()
                    img = image.load(img_path, 3, 'double')
                 end)

                 -- Sometimes image.load fails because the file extension does not match the
                 -- image format. In that case, use image.decompress on a ByteTensor.
                 if not ok then
                    local f = io.open(img_path, 'r')
                    assert(f, 'Error reading: ' .. tostring(img_path))
                    local data = f:read('*a')
                    f:close()

                    local b = torch.ByteTensor(string.len(data))
                    ffi.copy(b:data(), data, b:size(1))

                    img = image.decompress(b, 3, 'double')
                 end
            end
            if img:size(1) == 1 then 
              --dmn.logger:print("Converting grayscale image to color")
              new_img = torch.zeros(3, img:size(2), img:size(3))
              new_img[1] = img[1]
              new_img[2] = img[1]
              new_img[3] = img[1]
            elseif img:size(1) == 3 then 
              new_img = img
            else 
              local cur_size = img:size(1)
              error("Only operating on color or greyscale images " .. cur_size)
            end
            --dmn.logger:print("Successfully loaded image from " .. img_path)
          end,
          function(err)
            dmn.logger:print("ERROR OCCURED LOADING IMAGE from " .. img_path .. " " .. err)
            new_img = nil
          end)
  return new_img
end

function functions.file_exists(file_path)
  assert(file_path ~= nil, "Must specify file path to check")

  -- file exists if it has attributes
  local file_exists = lfs.attributes(file_path)
  return file_exists
end

-- check if folder exists
function functions.check_folder(base_dir)
  assert(base_dir ~= nil, "Must specify folder name to check")
  if lfs.attributes(base_dir) == nil then
    print("Directory not found, making new directory at " .. base_dir)
    lfs.mkdir(base_dir)
  end
end

-- executes command and returns all print lines
function functions.execute_command(command)
  assert(command ~= nil, "Must specify command to execute")
  local handle = io.popen(command)
  local result = handle:lines()

  handle:close()
  return result
end

-- Creates a new post request to specified url, returns body result
function functions.post_request(url, request_body)
  assert(url ~= nil, "Must specify url to send a POST request to")
  assert(request_body ~= nil, "Must specify body data to post")

  --dmn.logger:print("Sending POST request to " .. url
  --  .. " with data " .. tostring(request_body))

  local response_body = { }
  local res, code, response_headers = http.request
  {
      url = url;
      method = "POST";
      headers = 
      {
        ["Content-Type"] = "application/json";
        ["Content-Length"] = #request_body;
      };
      source = ltn12.source.string(request_body);
      sink = ltn12.sink.table(response_body);
  }

  assert(response_body ~= nil, "Error sending job id")
  return response_body

end

function functions.url_encode(str)
   assert(str ~= nil, "Must specify string to encode")
   if (str) then
      str = string.gsub (str, "\n", "\r\n")
      str = string.gsub (str, "([^%w ])",
         function (c) return string.format ("%%%02X", string.byte(c)) end)
      str = string.gsub (str, " ", "+")
   end
   return str    
end

function functions.trycatch(try,catch)
   local ok,err = pcall(try)
   if not ok then catch(err) end
end

-- returns a table of all files in a directory
function functions.list_files(dir)
  local files = {}
  local p = io.popen('find "'..dir..'" -type f')  --Open directory look for files, save data in p. By giving '-type f' as parameter, it returns all files.     
  for file in p:lines() do                         --Loop through all files
       table.insert(files, file)       
  end
  return files
end

-- returns file name of an absolute path
function functions.file_name(abs_path)
  assert(abs_path ~= nil, "Must specify absolute path to use")
  local dir_path, file_name, ext = 
  string.match(abs_path, "(.-)([^\\/]-%.?([^%.\\/]*))$")
  return file_name
end

function functions.http_request(url)
  assert(url ~= nil, "Must specify url to get")
  local r, c, headers = http.request(url)
  return r, c, headers
end

function functions.https_request(url)
  assert(url ~= nil, "Must specify https url to explore")
  local resp = {}

  https.TIMEOUT = 10
  local r, c, headers, s = https.request{
      url = url,
      sink = ltn12.sink.table(resp),
      protocol = "tlsv1"
  }

  local res = table.concat(resp)
  local trimmed_res = res:gsub("\\u","")
  local trimmed_res = trimmed_res:gsub("\\n", "")
  local trimmed_res = trimmed_res:gsub("@@", "")

  return res, r, headers
end