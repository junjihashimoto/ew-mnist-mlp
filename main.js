import { httpRequest } from 'http-request';
import { createResponse } from 'create-response';
import { logger } from 'log';
import { mlp, wrapImage } from './mlp.js';

export async function responseProvider(request) {
    let responseText = "";

    try {
        let origin = await httpRequest(`https://edgeworker-test0-gree-net.akamaized.net${request.path}`);
        let img = JSON.parse(await origin.text());
        let v = mlp(wrapImage(img));
        let str="";
        for(let i=0;i<28;i++){
            for(let j=0;j<28;j++){
                if (img.dat[i*28+j] < 0.2){
                    str+="_";
                }else{
                    str+="X";
                }
            }
            str+="</br>\n";
        }
        responseText = 
        '<html><body><h1>Hello MNIST From Akamai EdgeWorkers</h1>'+
        '<h2>' + str + '</h2>'+
        '<h2>Inference: ' + v.toString() + '</h2>'+
        ' </body></html>';
    } catch (error) {
        responseText = '<html><body><h1>Hello MNIST From Akamai EdgeWorkers</h1> <h2> Error </h2> </body></html>';
        logger.log("orign response errored: %s", error);
    } finally {
        logger.log("done processing");
    }
    return createResponse(
        200,
        {},
        responseText
    );
}
