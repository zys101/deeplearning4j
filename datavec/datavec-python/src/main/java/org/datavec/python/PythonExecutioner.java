/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.datavec.python;


import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.regex.Pattern;

import lombok.extern.slf4j.Slf4j;
import org.json.simple.JSONArray;
import org.json.simple.parser.JSONParser;
import org.bytedeco.javacpp.*;
import org.bytedeco.cpython.*;
import static org.bytedeco.cpython.global.python.*;
import org.nd4j.linalg.api.buffer.DataType;

/**
 *  Python executioner
 *
 *  @author Fariz Rahman
 */
@Slf4j
public class PythonExecutioner {
    private static Pointer namePtr;
    private static PyObject module;
    private static PyObject globals;
    private static JSONParser parser = new JSONParser();
    private static Map<String, PyThreadState> interpreters = new HashMap<String, PyThreadState>();
    private static String defaultInterpreter = "_main";
    private static String currentInterpreter =  defaultInterpreter;
    private static boolean currentInterpreterEnabled = false;
    private static boolean safeExecFlag = false;
    private static PyThreadState defaultThreadState;
    private static PyThreadState currentThreadState;
    private static long mainThreadId;
    private static String tempFile = "temp.json";

    static {
        init();
    }
    private static void setInterpreter(String name){
        if (name == null){ // switch to default interpreter
            currentInterpreter = defaultInterpreter;
            return;
        }

        if (!interpreters.containsKey(name)){
            log.info("CPython: Py_NewInterpreter()");
            interpreters.put(name, Py_NewInterpreter());
        }
        currentInterpreter = name;
    }

    private static void deleteInterpreter(String name){
        if (name == null || name == defaultInterpreter){
            return;
        }

        PyThreadState ts = interpreters.get(name);
        if (ts == null){
            return;
        }

        boolean isDeletingCurrentInterpreter = currentInterpreter == name;

        log.info("CPython: PyThreadState_Swap()");
        PyThreadState_Swap(ts);
        log.info("CPython: Py_EndInterpreter()");
        Py_EndInterpreter(ts);


        if (isDeletingCurrentInterpreter){
            currentInterpreter = defaultInterpreter;
        }
        log.info("CPython: PyThreadState_Swap()");
        PyThreadState_Swap(interpreters.get(defaultInterpreter));

    }


    public static void init(){
        System.out.println("---init()---");
        log.info("CPython: Py_DecodeLocale()");
        namePtr = Py_DecodeLocale("pythonExecutioner", null);
        log.info("CPython: Py_SetProgramName()");
        Py_SetProgramName(namePtr);
        log.info("CPython: Py_Initialize()");
        Py_Initialize();
        log.info("CPython: PyEval_InitThreads()");
        //PyEval_InitThreads();
        log.info("CPython: PyImport_AddModule()");
        module = PyImport_AddModule("__main__");
        log.info("CPython: PyModule_GetDict()");
        globals = PyModule_GetDict(module);
        log.info("CPython: PyThreadState_Get()");
        //interpreters.put(defaultInterpreter, PyThreadState_Get());
        System.out.println("---init()---Done-");
        mainThreadId = Thread.currentThread().getId();
    }

    public static void free(){
        log.info("CPython: Py_FinalizeEx()");
        if (Py_FinalizeEx() < 0) {
            throw new RuntimeException("Python execution failed.");
        }
        log.info("CPython: PyMem_RawFree()");
        PyMem_RawFree(namePtr);
        Py_Finalize();
    }


    private static String jArrayToPyString(Object[] array){
        String str = "[";
        for (int i=0; i < array.length; i++){
            Object obj = array[i];
            if (obj instanceof Object[]){
                str += jArrayToPyString((Object[])obj);
            }
            else if (obj instanceof String){
                str += "\"" + obj + "\"";
            }
            else{
                str += obj.toString().replace("\"", "\\\"");
            }
            if (i < array.length - 1){
                str += ",";
            }

        }
        str += "]";
        return str;
    }

    private static String escapeStr(String str){
        str = str.replace("\\", "\\\\");
        str = str.replace("\"\"\"", "\\\"\\\"\\\"");
        return str;
    }
    private static String inputCode(PythonVariables pyInputs)throws Exception{
        String inputCode = "loc={};";
        if (pyInputs == null){
            return inputCode;
        }
        Map<String, String> strInputs = pyInputs.getStrVariables();
        Map<String, Long> intInputs = pyInputs.getIntVariables();
        Map<String, Double> floatInputs = pyInputs.getFloatVariables();
        Map<String, NumpyArray> ndInputs = pyInputs.getNDArrayVariables();
        Map<String, Object[]> listInputs = pyInputs.getListVariables();
        Map<String, String> fileInputs = pyInputs.getFileVariables();

        String[] VarNames;


        VarNames = strInputs.keySet().toArray(new String[strInputs.size()]);
        for(Object varName: VarNames){
            String varValue = strInputs.get(varName);
            inputCode += varName + " = \"\"\"" + escapeStr(varValue) + "\"\"\"\n";
            inputCode += "loc['" + varName + "']=" + varName + "\n";
        }

        VarNames = intInputs.keySet().toArray(new String[intInputs.size()]);
        for(String varName: VarNames){
            Long varValue = intInputs.get(varName);
            inputCode += varName + " = " + varValue.toString() + "\n";
            inputCode += "loc['" + varName + "']=" + varName + "\n";
        }

        VarNames = floatInputs.keySet().toArray(new String[floatInputs.size()]);
        for(String varName: VarNames){
            Double varValue = floatInputs.get(varName);
            inputCode += varName + " = " + varValue.toString() + "\n";
            inputCode += "loc['" + varName + "']=" + varName + "\n";
        }

        VarNames = listInputs.keySet().toArray(new String[listInputs.size()]);
        for (String varName: VarNames){
            Object[] varValue = listInputs.get(varName);
            String listStr = jArrayToPyString(varValue);
            inputCode += varName + " = " + listStr + "\n";
            inputCode += "loc['" + varName + "']=" + varName + "\n";
        }

        VarNames = fileInputs.keySet().toArray(new String[fileInputs.size()]);
        for(Object varName: VarNames){
            String varValue = fileInputs.get(varName);
            inputCode += varName + " = \"\"\"" + escapeStr(varValue) + "\"\"\"\n";
            inputCode += "loc['" + varName + "']=" + varName + "\n";
        }

        if (ndInputs.size()> 0){
            inputCode += "import ctypes; import numpy as np;";
            VarNames = ndInputs.keySet().toArray(new String[ndInputs.size()]);

            String converter = "__arr_converter = lambda addr, shape, type: np.ctypeslib.as_array(ctypes.cast(addr, ctypes.POINTER(type)), shape);";
            inputCode += converter;
            for(String varName: VarNames){
                NumpyArray npArr = ndInputs.get(varName);
                npArr = npArr.copy();
                String shapeStr = "(";
                for (long d: npArr.getShape()){
                    shapeStr += String.valueOf(d) + ",";
                }
                shapeStr += ")";
                String code;
                String ctype;
                if (npArr.getDtype() == DataType.FLOAT){

                    ctype = "ctypes.c_float";
                }
                else if (npArr.getDtype() == DataType.DOUBLE){
                    ctype = "ctypes.c_double";
                }
                else if (npArr.getDtype() == DataType.SHORT){
                    ctype = "ctypes.c_int16";
                }
                else if (npArr.getDtype() == DataType.INT){
                    ctype = "ctypes.c_int32";
                }
                else if (npArr.getDtype() == DataType.LONG){
                    ctype = "ctypes.c_int64";
                }
                else{
                    throw new Exception("Unsupported data type: " + npArr.getDtype().toString() + ".");
                }

                code = "__arr_converter(" + String.valueOf(npArr.getAddress()) + "," + shapeStr + "," + ctype + ")";
                code = varName + "=" + code + "\n";
                inputCode += code;
                inputCode += "loc['" + varName + "']=" + varName + "\n";
            }

        }
        return inputCode;
    }

    private String outputCode(PythonVariables pyOutputs){
        if (pyOutputs == null){
            return "";
        }
        String outputCode = "import json;json.dump({";
        String[] VarNames = pyOutputs.getVariables();
        boolean ndarrayHelperAdded = false;
        for (String varName: VarNames){
            if (pyOutputs.getType(varName) == PythonVariables.Type.NDARRAY){
                if (! ndarrayHelperAdded){
                    ndarrayHelperAdded = true;
                    String helper = "serialize_ndarray_metadata=lambda x:{\"address\":x.__array_interface__['data'][0]" +
                            ",\"shape\":x.shape,\"strides\":x.strides,\"dtype\":str(x.dtype)};";
                    outputCode = helper + outputCode;
                }
                outputCode += "\"" + varName + "\"" + ":serialize_ndarray_metadata(" + varName + "),";

            }
            else {
                outputCode += "\"" + varName + "\"" + ":" + varName + ",";
            }
        }
        outputCode = outputCode.substring(0, outputCode.length() - 1);
        outputCode += "},open('" + tempFile + "', 'w'));";
        return outputCode;

    }

    private static void _readOutputs(PythonVariables pyOutputs){
        if (pyOutputs == null){
            return;
        }
        /*
        exec(getOutputCheckCode(pyOutputs));
        String errorMessage = evalSTRING("__error_message");
        if (errorMessage.length() > 0){
            throw new RuntimeException(errorMessage);
        }*/
        try{

            for (String varName: pyOutputs.getVariables()){
                PythonVariables.Type type = pyOutputs.getType(varName);
                if (type == PythonVariables.Type.STR){
                    pyOutputs.setValue(varName, evalSTRING(varName));
                }
                else if(type == PythonVariables.Type.FLOAT){
                    pyOutputs.setValue(varName, evalFLOAT(varName));
                }
                else if(type == PythonVariables.Type.INT){
                    pyOutputs.setValue(varName, evalINTEGER(varName));
                }
                else if (type == PythonVariables.Type.LIST){
                    Object varVal[] = evalLIST(varName);
                    pyOutputs.setValue(varName, varVal);
                }
                else{
                    pyOutputs.setValue(varName, evalNDARRAY(varName));
                }
            }
        }
        catch (Exception e){
            log.error(e.toString());
        }

    }

    private static void _enterSubInterpreter() {
        if (!currentInterpreterEnabled && currentInterpreter != defaultInterpreter) {


            //log.info("CPython: PyEval_AcquireLock()");



            PyThreadState ts = interpreters.get(currentInterpreter);
            if (Thread.currentThread().getId() == mainThreadId){
                PyThreadState_Swap(ts);
            }
            else{
                System.out.println("call from thread: " + Thread.currentThread().getId());
                defaultThreadState = PyEval_SaveThread();
                log.info("CPython: PyThreadState.interp()");

                PyInterpreterState is = ts.interp();
                log.info("CPython: PyThreadState_New()");
                ts = PyThreadState_New(is);
                log.info("CPython: PyThreadState_Swap()");
                PyThreadState_Swap(ts);

                PyEval_RestoreThread(ts);




            }
            currentThreadState = ts;
            currentInterpreterEnabled = true;

            System.out.println("--enter done---");
        }
    }

    private static void _exitSubInterpreter(){
        if (currentInterpreterEnabled && currentInterpreter != defaultInterpreter){


            if (Thread.currentThread().getId() == mainThreadId){
                PyThreadState_Swap(null);

            }
            else{
                System.out.println("call from thread: " + Thread.currentThread().getId());
                PyThreadState ts = currentThreadState;
                log.info("CPython: PyThreadState_Swap()");
                PyThreadState_Swap(null);

                PyEval_RestoreThread(defaultThreadState);
                System.out.println("--exit done--");
                log.info("CPython: PyThreadState_Clear()");
                PyThreadState_Clear(ts);
                log.info("CPython: PyThreadState_Delete()");
                PyThreadState_Delete(ts);
                log.info("CPython: PyEval_ReleaseLock()");

            }

            currentInterpreterEnabled = false;
            System.out.println("--exit done--");

        }

    }

    /**
     * Executes python code. Also manages python thread state.
     * @param code
     */
    public static void exec(String code){
        log.info("CPython: PyRun_SimpleStringFlag()");
        log.info(code);
        System.out.println("about to exec");
        PyRun_SimpleStringFlags(code, null);
        System.out.println("exec done");
        log.info("Exec done");
    }

    public static void exec(String code, PythonVariables pyOutputs){


        Object[] ndArrayOuts = pyOutputs.getNDArrayVariables().keySet().toArray();
        if (ndArrayOuts.length > 0){
            if (code.charAt(code.length() - 1) != '\n'){
                code += "\n";
            }

            String[] varNames = Arrays.copyOf(ndArrayOuts, ndArrayOuts.length, String[].class);
            for (String varName: varNames){
                code += getNDArrayOutCode(varName);
            }
        }
        exec(code);
        _readOutputs(pyOutputs);
        System.out.println("read done");

    }

    public static void exec(String code, String interpreter){
        setInterpreter(interpreter);
        _enterSubInterpreter();
        exec(code);
        _exitSubInterpreter();
    }

    public static void exec(String code, PythonVariables pyOutputs, String interpreter){
        setInterpreter(interpreter);
        _enterSubInterpreter();

        exec(code, pyOutputs);

        _exitSubInterpreter();
    }

    public static void exec(String code, PythonVariables pyInputs, PythonVariables pyOutputs) throws Exception{
        String inputCode = inputCode(pyInputs);
        if (code.charAt(code.length() - 1) != '\n'){
            code += '\n';
        }
        exec(inputCode + code, pyOutputs);
    }

    public static void exec(String code, PythonVariables pyInputs, PythonVariables pyOutputs, String interpreter) throws Exception{
        String inputCode = inputCode(pyInputs);
        if (code.charAt(code.length() - 1) != '\n'){
            code += '\n';
        }
        exec(inputCode + code, pyOutputs, interpreter);
    }

    private static void setupTransform(PythonTransform transform){
        setInterpreter(transform.getName());
    }
    public static PythonVariables exec(PythonTransform transform) throws Exception{
        setupTransform(transform);
        if (transform.getInputs() != null && transform.getInputs().getVariables().length > 0){
            throw new Exception("Required inputs not provided.");
        }
        exec(transform.getCode(), null, transform.getOutputs());
        return transform.getOutputs();
    }

    public static PythonVariables exec(PythonTransform transform, PythonVariables inputs)throws Exception{
        setupTransform(transform);
        exec(transform.getCode(), inputs, transform.getOutputs());
        return transform.getOutputs();
    }


    public static String evalSTRING(String varName){
//        module = PyImport_AddModule("__main__");
//        log.info("CPython: PyModule_GetDict()");
//        globals = PyModule_GetDict(module);
        PyObject xObj = PyDict_GetItemString(globals, varName);
        PyObject bytes = PyUnicode_AsEncodedString(xObj, "UTF-8", "strict");
        BytePointer bp = PyBytes_AsString(bytes);
        String ret = bp.getString();
        Py_DecRef(xObj);
        Py_DecRef(bytes);
        return ret;
    }

    public static long evalINTEGER(String varName){
//        module = PyImport_AddModule("__main__");
//        log.info("CPython: PyModule_GetDict()");
//        globals = PyModule_GetDict(module);
        PyObject xObj = PyDict_GetItemString(globals, varName);
        long ret = PyLong_AsLongLong(xObj);
        return ret;
    }

    public static double evalFLOAT(String varName){
//        module = PyImport_AddModule("__main__");
//        log.info("CPython: PyModule_GetDict()");
//        globals = PyModule_GetDict(module);
        PyObject xObj = PyDict_GetItemString(globals, varName);
        double ret = PyFloat_AsDouble(xObj);
        return ret;
    }

    public static Object[] evalLIST(String varName) throws Exception{
//        module = PyImport_AddModule("__main__");
//        log.info("CPython: PyModule_GetDict()");
//        globals = PyModule_GetDict(module);
        PyObject xObj = PyDict_GetItemString(globals, varName);
        PyObject strObj = PyObject_Str(xObj);
        PyObject bytes = PyUnicode_AsEncodedString(strObj, "UTF-8", "strict");
        BytePointer bp = PyBytes_AsString(bytes);
        String listStr = bp.getString();
        Py_DecRef(xObj);
        Py_DecRef(bytes);
        JSONArray jsonArray = (JSONArray)parser.parse(listStr.replace("\'", "\""));
        return jsonArray.toArray();
    }

    private static long[] longArrayFromTupleString(String str){
        str = str.replace(" ", "");
        str = str.substring(1, str.length() - 1);
        String[] items = str.split(Pattern.quote(","));
        long[] nums = new long[items.length];
        for (int i=0; i<nums.length; i++){
            nums[i] = Long.parseLong(items[i]);
        }
        return nums;
    }

    private static String getNDArrayOutCode(String varName){
        String code = "__%s_address = %s.__array_interface__['data'][0]\n" +
                "__%s_shape = str(%s.shape)\n" +
                "__%s_strides = str(%s.strides)\n" +
                "__%s_dtype = str(%s.dtype)\n";
        code = String.format(
                code,
                varName, varName, varName,
                varName, varName, varName,
                varName, varName
        );

        return code;
    }
    public static NumpyArray evalNDARRAY(String varName) throws Exception{
        System.out.println("evalNDArray()");





        long address = evalINTEGER(String.format("__%s_address", varName));
        System.out.println(address);
        long [] shape = longArrayFromTupleString(evalSTRING(String.format("__%s_shape", varName)));

        long [] strides = longArrayFromTupleString(evalSTRING(String.format("__%s_strides", varName)));
        String dtypeName = evalSTRING(String.format("__%s_dtype", varName));

        System.out.println(dtypeName);



        DataType dtype;
        if (dtypeName.equals("float64")){
            dtype = DataType.DOUBLE;
        }
        else if (dtypeName.equals("float32")){
            dtype = DataType.FLOAT;
        }
        else if (dtypeName.equals("int16")){
            dtype = DataType.SHORT;
        }
        else if (dtypeName.equals("int32")){
            dtype = DataType.INT;
        }
        else if (dtypeName.equals("int64")){
            dtype = DataType.LONG;
        }
        else{
            throw new Exception("Unsupported array type " + dtypeName + ".");
        }
        NumpyArray ret = new NumpyArray(address, shape, strides, dtype, safeExecFlag);


        /*
        Py_DecRef(arrayInterface);
        Py_DecRef(data);
        Py_DecRef(zero);
        Py_DecRef(addressObj);
        Py_DecRef(shapeObj);
        Py_DecRef(stridesObj);
        */


        System.out.println("evalNDArray()-Done");
       return ret;
    }

    private static String getOutputCheckCode(PythonVariables pyOutputs){
        // make sure all outputs exist and are of expected types
        // helps avoid JVM crashes (most of the time)
        String code= "__error_message=''\n";
        String checkVarExists = "if '%s' not in locals(): __error_message += '%s not found.'\n";
        String checkVarType = "if not isinstance(%s, %s): __error_message += '%s is not of required type.'\n";
        for (String varName: pyOutputs.getVariables()){
            PythonVariables.Type type = pyOutputs.getType(varName);
            code += String.format(checkVarExists, varName, varName);
            switch(type){
                case INT:
                    code += String.format(checkVarType, varName, "int", varName);
                    break;
                case STR:
                    code += String.format(checkVarType, varName, "str", varName);
                    break;
                case FLOAT:
                    code += String.format(checkVarType, varName, "float", varName);
                    break;
                case BOOL:
                    code += String.format(checkVarType, varName, "bool", varName);
                    break;
                case NDARRAY:
                    code += String.format(checkVarType, varName, "np.ndarray", varName);
                    break;
                case LIST:
                    code += String.format(checkVarType, varName, "list", varName);
                    break;
            }
        }
        return code;
    }
}
