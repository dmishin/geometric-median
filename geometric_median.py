import numpy as np
from PIL import Image
import os
import time
from contextlib import contextmanager
from multiprocessing import cpu_count
from queue import Queue, Empty
from threading import Thread

@contextmanager
def print_time( formatstr="Execution time: {time:0.3}s" ):
    t0 = time.time()
    yield
    t1 = time.time()
    report( formatstr.format(time = t1-t0) )

__verbose = True
def report(*args, **kwargs):
    if __verbose:
        print(*args, **kwargs)    

        
def load(path):
    data = np.asarray(Image.open(path))
    w,h,d = data.shape
    if d == 4:
        #report("Image has 4 channels, adduming channel #4 is alpha and dropping it")
        data = data[:,:,0:3]
    return data.astype(np.float32)

def show_rgbimage( rgb ):
    image = Image.fromarray( rgb.astype(np.uint8) )
    if os.name=="nt":
        name = "image.jpg"
        image.save(name)
        os.system("start {}".format(name))
        print ("Saved", name)
    else:
        image.show()
        
def weighted_image( img_ref, image_loader, y, bias, norm ):
    #calculates weight and return premultipleit image and weight
    img = image_loader(img_ref)
    w,h, _ = img.shape
    #print ("#### norm order is", norm, "bias is", bias)
    if norm == 2:
        dimg = np.square(img-y).sum(axis=2, keepdims=True)
        dimg = np.sqrt(dimg, out=dimg)
    elif norm == 1:
        dimg = np.abs(img-y).sum(axis=2, keepdims=True)
    elif norm == np.inf:
        dimg = np.abs(img-y).max(axis=2, keepdims=True)
    else:
        raise ValueError("Unsupported order")
    
    dimg += bias
    dimg = np.reciprocal(dimg, out=dimg)    
    return img*dimg, dimg

def geommean_worker(imageQueue, imageLoader, imageAccumulator, weightAccumulator, baseImage, bias, norm, name="thread"):
    try:
        while True:
            #print ("##### {name}: Trying to process image...".format(**locals()))
            wimage, weight = weighted_image(imageQueue.get_nowait(),
                                            imageLoader,
                                            baseImage,
                                            bias,
                                            norm)
            #print ("##### {name}: Done processing, accumulating".format(**locals()))
            imageAccumulator += wimage
            weightAccumulator += weight
            imageQueue.task_done()
            #print ("##### {name}: Done accumulating".format(**locals()))            
    except Empty:
        #print("Thread {name} terminated".format(**locals()))
        pass
    except Exception as err:
        print("Unexpected termination of {name}: {err}".format(**locals()))

def mean_worker(imageQueue, imageLoader, imageAccumulator, name="thread"):
    try:
        while True:
            imageAccumulator += imageLoader( imageQueue.get_nowait() )
            imageQueue.task_done()
    except Empty:
        pass
        
def geommean_images(images, image_loader, iters = 100, bias=1e-6, threads=None, norm=2):
    if threads is None: threads = cpu_count()+1
    n = len(images)
    
    print ("Calculating average of {n} images...".format(n=n))
    with print_time():
        y = mean_list( images, image_loader, threads )

    w,h,chans = y.shape

    for i in range(iters):
        print ("Iteration {i}/{n}".format(i=i+1, n=iters))
        with print_time():
            simg = np.zeros( shape=y.shape, dtype=np.float32 )
            sweig = np.zeros( shape=(w,h,1), dtype=np.float32 )

            imageQueue = Queue()
            for img in images:
                imageQueue.put(img)

            #run workers
            for i in range(threads):
                #geommean_worker(imageQueue, imageLoader, imageAccumulator, weightAccumulator, baseImage, bias, norm):
                worker = Thread(target = lambda: geommean_worker(imageQueue, image_loader,
                                                                 imageAccumulator = simg,
                                                                 weightAccumulator = sweig,
                                                                 baseImage = y,
                                                                 bias=bias,
                                                                 norm=norm,
                                                                 name="worker {}".format(i)))
                worker.start()
                worker = None
            imageQueue.join()
                                
            #map of distances for each point is done
            simg /= sweig
            y = simg

    return y


def mean_list( images, loadImage, threads ):
    num = 0
    base = loadImage(images[0]).copy()
    imageQueue = Queue()
    for img in images[1:]:
        imageQueue.put(img)
    for i in range(threads):
        #def mean_worker(imageQueue, imageLoader, imageAccumulator, name="thread"):
        worker = Thread(target = lambda: mean_worker(imageQueue, loadImage,
                                                     base,
                                                     name="thread {}".format(i)))
        worker.start()
    imageQueue.join()
    base *= (1.0 / len(images))
    return base
    
def mean_list1( matrices ):
    imatrix = iter(matrices)
    sm = next(imatrix).copy()
    sw = 1.0
    for m in imatrix:
        sm += m
        sw += 1.0
    sm *= (1/sw)
    return sm

class BaseColorMap:
    def mapped_loader( self, loader ):
        def modified_loader( arg ):
            return self.map(loader(arg))
        return modified_loader        
        
    def map(self, image): assert False
    def unmap(self, mimage): assert False
    
class DirectMap(BaseColorMap):
    def __init__(self, *args): pass
    def mapped_loader(self, loader): return loader
    def map(self, image): return image
    def unmap(self, mimage): return mimage
    
class ProjectiveColorMap(BaseColorMap):
    def __init__(self, str_luma_scale=None):
        if str_luma_scale is None:
            luma_scale=0.3/(255.0*3)
        else:
            luma_scale = float(str_luma_scale)/(255.0*3)
        
        self.luma_scale = luma_scale
        self.bias = 1e-3
        
    def map(self, image):
        if False:
            luma = image[:,:,0:1].copy()
            luma += image[:,:,1:2]
            luma += image[:,:,2:3]
            luma += self.bias
            return np.dstack( (image / luma, luma*self.luma_scale) )
        
        if True:
            w,h,d = image.shape
            output = np.zeros((w,h,d+1), dtype = np.float32)
            luma = output[:,:,d:d+1]
            luma += self.bias
            for i in range(d):
                luma[:] +=  image[:,:,i:i+1]

            output[:,:,0:d] = image
            output[:,:,0:d] /= luma
            luma *= self.luma_scale
            return output
    
    def unmap(self, mimage):
        luma = mimage[:,:,3:4] * (1/self.luma_scale)
        luma -= self.bias
        return mimage[:,:,0:3] * luma

def main():
    import sys
    from argparse import ArgumentParser
    
    parser = ArgumentParser(description="Calculate geometrical mean of the set of images")

    parser.add_argument("-g", "--glob", dest="glob", action="store_true",
                        default=False,
                        help="Apply glob on given paths (useful on windows)")
    
    parser.add_argument("-p", "--preload", dest="preload", action="store_true",
                        default=False,
                        help="Preload all images to memory. Faster, but requires a lot of memory.")

    parser.add_argument("-t", "--threads", dest="threads", type=int, metavar="N",
                        help="Use N worker threads, default is processor number+1")
                      
    parser.add_argument("-i", "--iterations", dest="iterations", type=int,
                        metavar="N", default = 10,
                        help="Number of iterations of the algorithm")

    parser.add_argument("--bias", dest="bias", type=float,
                        metavar="BIAS", default = 1e-6,
                        help="Bias of the algorithm, small positive floating-point value. Default is 1e-6.")
    
    parser.add_argument("-n", "--norm", dest="norm",
                        metavar="ORDER", default = "2",
                        help="Order of the norm function to calculate distances. Possible values are 1, 2, inf, -inf")

    parser.add_argument("-c", "--colormap", dest="colormap",
                        metavar="NAME", default = "none",
                        help="Apply specified color mapping before averaging images. Possible values are: none (use RGB as is), projective[:luma_weight=0.3] (suppress importance of luminance by luma_weight factor, value between 0 and 1)")
    
    parser.add_argument("-o", "--output",
                        dest="output",
                        help="Output image file")
    
    parser.add_argument("inputs", nargs="+", metavar="IMAGE",
                        help="Input image")
    
    options = parser.parse_args()
    if not options.glob:
        files = options.inputs
    else:
        import glob
        files = []
        for pattern in options.inputs:
            files.extend(glob.glob(pattern))
            
    if len(files) < 1:
        parser.error("No input files")
            
    for filename in files:
        if not os.path.exists(filename):
            parser.error("File {filename} not found".format(**locals()))

    if options.preload:
        images = [load(filename) for filename in files]
        image_loader = lambda x: x
        print( "Preloaded {n} images".format(n=len(images)))
    else:
        images = files
        image_loader = load

    try:
        order = {"1": 1, "2": 2,"inf": np.inf,}[options.norm]
    except KeyError:
        parser.error("Wrong norm order: {}".format(options.norm))

    try:
        mappers = {"none": DirectMap,
                  "projective": ProjectiveColorMap}
        cmap_args = options.colormap.split(':')
        
        mapper = mappers[cmap_args[0].lower()](*cmap_args[1:])
        
    except KeyError:
        parser.error("Wrong colormap name: {name}, available names are: {names}".format(name = cmap_args[0], names = ", ".join(sorted(mappers.keys()))))
    except Exception as err:
        parser.error("Failed to initialize colormap {name} with arguments {args}, error: {err}".format(name=cmap_args[0], args=", ".join(cmap_args[1:]), err=err))
        
    y = geommean_images(images, mapper.mapped_loader(image_loader), iters=options.iterations, bias=options.bias, threads=options.threads, norm=order)
    del images
    y = mapper.unmap(y)
    y = np.clip(y, 0.0, 255.0, out=y)
    if not options.output:
        show_rgbimage(y)
    else:
        image = Image.fromarray( y.astype(np.uint8) )
        image.save(options.output)
        print ("Saved", options.output)
        
if __name__=="__main__":
    main()
