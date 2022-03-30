import { model } from './model.js';

function rescaleArray (a) {
//    let len = Object.keys(a.dat).length;
    let len = a.dat.length;
    let d = new Float32Array(len);
    for(let i=0;i<len;i++){
	d[i] = a.dat[i]/10000.0;
    }
    a.dat = d;
}
function rescale (m) {
    rescaleArray(m.w0);
    rescaleArray(m.b0);
    rescaleArray(m.w1);
    rescaleArray(m.b1);
    rescaleArray(m.w2);
    rescaleArray(m.b2);
}

function readArray (file,shape){
    let w = fs.readFileSync(file);
    let b = new Float32Array(w.length/4);
    for(let i=0;i<b.length;i++)
	b[i] = w.readFloatLE(i*4);
    return {
	dat : b,
	shape : shape
    };
}

export function wrapImage(a){
    let v = {
	dat : new Float32Array(a.shape[0]),
	shape : [a.shape[0]]
    };
    for(let i=0;i<a.shape[0];i++){
	v.dat[i] = (a.dat[i] / 255) - 0.5;
    }
    return v;
}

function relu(a){
    let v = {
	dat : new Float32Array(a.shape[0]),
	shape : [a.shape[0]]
    };
    for(let i=0;i<a.shape[0];i++){
	v.dat[i] = a.dat[i] >= 0 ? a.dat[i] : 0;
    }
    return v;
}

function softmax(a){
    let v = {
	dat : new Float32Array(a.shape[0]),
	shape : [a.shape[0]]
    };
    let t = 0.0001;
    for(let i=0;i<a.shape[0];i++){
	v.dat[i] = Math.exp(a.dat[i]);
	t+=v.dat[i];
    }
    for(let i=0;i<a.shape[0];i++){
	v.dat[i] /= t;
    }
    return v;
}

function argmax(a) {
    let t = a.dat[0];
    let max = 0;
    for(let i=1;i<a.shape[0];i++){
	if(t<a.dat[i]){
	    t = a.dat[i];
	    max = i;
	}
    }
    return max;
}

function ax_b(a,x,b){
    let v = {
	dat : new Float32Array(a.shape[0]),
	shape : [a.shape[0]]
    };
    for(let i=0;i<a.shape[0];i++){
	let t = 0;
	let offset = i*a.shape[1];
	for(let j=0;j<a.shape[1];j++){
	    t += a.dat[offset+j] * x.dat[j];
	}
	v.dat[i] = t;
    }
    for(let i=0;i<a.shape[0];i++){
	v.dat[i] += b.dat[i];
    }
    return v;
}

function readModel (){
    return {
	w0: readArray('l0.w',[64,784]),
	b0: readArray('l0.b',[64]),
	w1: readArray('l1.w',[32,64]),
	b1: readArray('l1.b',[32]),
	w2: readArray('l2.w',[10,32]),
	b2: readArray('l2.b',[10])
    };
}

export function mlp(img) {
    let x0 = relu(ax_b(model.w0,img,model.b0))
    let x1 = relu(ax_b(model.w1,x0,model.b1))
    let x2 = softmax(ax_b(model.w2,x1,model.b2))
    return argmax(x2);
}

// 784 64 32 10
rescale(model);

