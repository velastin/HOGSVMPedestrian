// (c) Diego Gomez 2015, USACH
// Adapted by S.A. Velastin 2016, UC3M

#include "functions.hpp"

void readXMLFile(string input_arch, deque<deque<Cuerpo> >& cuerpos, deque<deque<Estado> >& estados, unsigned int &total_frames){
	TiXmlDocument doc;
	if(!doc.LoadFile(input_arch.c_str())){
		cerr << "File couldn't be opened: '" << input_arch << "'" << endl;
		exit(0);
	}
	TiXmlElement* root = doc.FirstChildElement();
    string id;

	for(TiXmlElement* elem = root->FirstChildElement(); elem != NULL; elem = elem->NextSiblingElement()){
    	string elemName = elem->Value();
    	if(elemName == "data"){
    		for(TiXmlElement* elem2 = elem->FirstChildElement(); elem2 != NULL; elem2 = elem2->NextSiblingElement()){
    			elemName = elem2->Value(); 				
    			if(elemName == "sourcefile"){
    				for(TiXmlElement* elem3 = elem2->FirstChildElement(); elem3 != NULL; elem3 = elem3->NextSiblingElement()){
    					elemName = elem3->Value();
    					if(elemName == "object"){
    						deque<Cuerpo> cuerpo_aux;
    						deque<Estado> estado_aux;
                            id = elem3->Attribute("id");
                            for(TiXmlElement* elem4 = elem3->FirstChildElement(); elem4 != NULL; elem4 = elem4->NextSiblingElement()){
                                if((elemName = elem4->Attribute("name"))=="Cuerpo"){
    								for(TiXmlElement* elem5 = elem4->FirstChildElement(); elem5 != NULL; elem5 = elem5->NextSiblingElement()){
    									Cuerpo cuerpo;
    									stringstream s;
    									char* se = const_cast<char*>(elem5->Attribute("framespan"));
    									se = strtok(se, ":");
    									cuerpo.inicio = atoi(se);
    									cuerpo.fin = atoi(strtok(NULL, ":"));
    									s << id << " " << elem5->Attribute("x") << " " << elem5->Attribute("y") << " " << elem5->Attribute("width") << " " << elem5->Attribute("height");
    									s >> cuerpo.id >> cuerpo.x >> cuerpo.y >> cuerpo.w >> cuerpo.h;
    									cuerpo_aux.push_back(cuerpo);
    								}
    							}else if((elemName = elem4->Attribute("name")) == "Estado"){
    								for(TiXmlElement* elem5 = elem4->FirstChildElement(); elem5 != NULL; elem5 = elem5->NextSiblingElement()){
	    								Estado estado;
	    								stringstream s;
	    								char* se = const_cast<char*>(elem5->Attribute("framespan"));
	    								se = strtok(se, ":");
	    								estado.inicio = atoi(se);
	    								estado.fin = atoi(strtok(NULL, ":"));
										s << id << " " << elem5->Attribute("value");
										s >> estado.id >> estado.estado;
										estado_aux.push_back(estado);
	    							}
    							}
    						}
    						cuerpos.push_back(cuerpo_aux);
    						estados.push_back(estado_aux);
    					}else if(elemName == "file"){
                            for(TiXmlElement* elem4 = elem3->FirstChildElement(); elem4 != NULL; elem4 = elem4->NextSiblingElement()){
                                if((elemName = elem4->Attribute("name"))=="NUMFRAMES"){
                                    for(TiXmlElement* elem5 = elem4->FirstChildElement(); elem5 != NULL; elem5 = elem5->NextSiblingElement()){
                                        char* se = const_cast<char*>(elem5->Attribute("value"));
                                        total_frames = atoi(se);
                                    }
                                }
                            }
                        }
    				}
    			}
    		}
    	}
    }
}

bool checkNear(float percentage, Rect rec, int pos, deque<deque<Cuerpo> >& cuerpos, int virtualFrame, int num_bodies){
    double area, area_aux;
    Rect next_body_rec;
    for(int i = 0; i < num_bodies; i++){
        if(i != pos){
            if(cuerpos.at(i).size() > 0){
                if(virtualFrame >= cuerpos.at(i).at(0).inicio){
                    if(virtualFrame <= cuerpos.at(i).at(0).fin){
                        next_body_rec = Rect(cuerpos.at(i).at(0).x, cuerpos.at(i).at(0).y, cuerpos.at(i).at(0).w, cuerpos.at(i).at(0).h);
                        area = rec.area();
                        Rect aux = next_body_rec & rec;
                        area_aux = aux.area();
                        if(area_aux >= area * percentage){
                            return false;              
                        }
                    }else{
                        if(cuerpos.at(i).size() > 1){
                            if(virtualFrame >= cuerpos.at(i).at(1).inicio){
                                if(virtualFrame <= cuerpos.at(i).at(1).fin){
                                    next_body_rec = Rect(cuerpos.at(i).at(1).x, cuerpos.at(i).at(1).y, cuerpos.at(i).at(1).w, cuerpos.at(i).at(1).h);
                                    Rect aux = rec & next_body_rec;
                                    area = rec.area();
                                    area_aux = aux.area();
                                    if(area_aux >= area * percentage){
                                        return false;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    return true;
}

bool checkNegative(Rect &neg_rec, int pos, deque<deque<Cuerpo> >& cuerpos, int virtualFrame, int num_bodies, int height, int width, Size size){
    double area, area_aux;
    Cuerpo negative;
    int rounds = 0;
    bool inter = true;
    while(inter && rounds < 5){
        inter = false;
        negative.x = rand() % (size.width - width);
        negative.y = rand() % (size.height - height);
        if((negative.x + width) > size.width) negative.x = negative.x - width;
        if((negative.y + height) > size.height) negative.y = negative.y - height;
        neg_rec = Rect(negative.x, negative.y, width, height);
        for(int k = 0; k < num_bodies; k++){
            if(cuerpos.at(k).size() > 0){
                if(virtualFrame >= cuerpos.at(k).at(0).inicio){
                    if(virtualFrame <= cuerpos.at(k).at(0).fin){
                        Rect rec_pos(cuerpos.at(k).at(0).x, cuerpos.at(k).at(0).y, cuerpos.at(k).at(0).w, cuerpos.at(k).at(0).h);
                        Rect aux = rec_pos & neg_rec;
                        area = neg_rec.area();
                        area_aux = aux.area();
                        if(area_aux >= area * 0.1){
                            inter = true;
                            break;
                        }
                    }else{
                        if(cuerpos.at(k).size() > 1){
                            if(virtualFrame >= cuerpos.at(k).at(1).inicio){
                                if(virtualFrame <= cuerpos.at(k).at(1).fin){
                                    Rect rec_pos(cuerpos.at(k).at(1).x, cuerpos.at(k).at(1).y, cuerpos.at(k).at(1).w, cuerpos.at(k).at(1).h);
                                    Rect aux = rec_pos & neg_rec;
                                    area = neg_rec.area();
                                    area_aux = aux.area();
                                    if(area_aux >= area * 0.1){
                                        inter = true;
                                        break;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        rounds++;
    } 
    return inter;
}

