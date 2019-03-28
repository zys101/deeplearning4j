/*******************************************************************************
 * Copyright (c) 2015-2019 Skymind, Inc.
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

package org.datavec.api.transform.transform.categorical;

import org.datavec.api.transform.metadata.CategoricalMetaData;
import org.datavec.api.transform.metadata.ColumnMetaData;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.transform.transform.BaseTransform;
import org.datavec.api.writable.Text;
import org.datavec.api.writable.Writable;
import org.nd4j.base.Preconditions;
import org.nd4j.shade.jackson.annotation.JsonIgnoreProperties;
import org.nd4j.shade.jackson.annotation.JsonProperty;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

@JsonIgnoreProperties({"inputSchema", "columnIdx"})
public class FirstDigitTransform extends BaseTransform {
    public static final String OTHER_CATEGORY = "Other";

    public enum Mode {
        EXCEPTION_ON_INVALID,
        INCLUDE_OTHER_COLUMN
    }

    protected String inputColumn;
    protected String outputColumn;
    protected Mode mode;
    private int columnIdx = -1;

    public FirstDigitTransform(@JsonProperty("inputColumn") String inputColumn, @JsonProperty("outputColumn") String outputColumn,
                               @JsonProperty("mode") Mode mode){
        this.inputColumn = inputColumn;
        this.outputColumn = outputColumn;
        this.mode = mode;
    }

    @Override
    public List<Writable> map(List<Writable> writables) {
        List<Writable> out = new ArrayList<>();
        int i=0;
        boolean inplace = inputColumn.equals(outputColumn);
        for(Writable w : writables){
            if(i++ == columnIdx) {
                if(!inplace){
                    out.add(w);
                }

                String s = w.toString();
                if (s.isEmpty()) {
                    if (mode == Mode.INCLUDE_OTHER_COLUMN) {
                        out.add(new Text(OTHER_CATEGORY));
                    } else {
                        throw new IllegalStateException("Encountered empty string in FirstDigitTransform that is set to Mode.EXCEPTION_ON_INVALID." +
                                " Either data contains an invalid (non-numerical) entry, or set FirstDigitTransform to Mode.INCLUDE_OTHER_COLUMN");
                    }
                } else {
                    char first = s.charAt(0);
                    if (first == '-' && s.length() > 1) {
                        //Handle negatives
                        first = s.charAt(1);
                    }
                    if (first >= '0' && first <= '9') {
                        out.add(new Text(String.valueOf(first)));
                    } else {
                        if (mode == Mode.INCLUDE_OTHER_COLUMN) {
                            out.add(new Text(OTHER_CATEGORY));
                        } else {
                            String s2 = s;
                            if (s.length() > 100) {
                                s2 = s2.substring(0, 100) + "...";
                            }
                            throw new IllegalStateException("Encountered string \"" + s2 + "\" with non-numerical first character in " +
                                    "FirstDigitTransform that is set to Mode.EXCEPTION_ON_INVALID." +
                                    " Either data contains an invalid (non-numerical) entry, or set FirstDigitTransform to Mode.INCLUDE_OTHER_COLUMN");
                        }
                    }
                }
            } else {
                out.add(w);
            }
        }
        return out;
    }

    @Override
    public Object map(Object input) {
        throw new UnsupportedOperationException("Not supported");
    }

    @Override
    public Object mapSequence(Object sequence) {
        throw new UnsupportedOperationException("Not supported");
    }

    @Override
    public String toString() {
        return "FirstDigitTransform(input=\"" + inputColumn + "\",output=\"" + outputColumn + "\",mode=" + mode + ")";
    }

    @Override
    public Schema transform(Schema inputSchema) {
        List<String> origNames = inputSchema.getColumnNames();
        List<ColumnMetaData> origMeta = inputSchema.getColumnMetaData();

        Preconditions.checkState(origNames.contains(inputColumn), "Input column with name \"%s\" not found in schema", inputColumn);
        Preconditions.checkState(inputColumn.equals(outputColumn) || !origNames.contains(outputColumn),
                "Output column with name \"%s\" already exists in schema (only allowable if input column == output column)", outputColumn);

        List<ColumnMetaData> outMeta = new ArrayList<>(origNames.size()+1);
        for( int i=0; i<origNames.size(); i++ ){
            String s = origNames.get(i);
            if(s.equals(inputColumn)){
                if(!outputColumn.equals(inputColumn)){
                    outMeta.add(origMeta.get(i));
                }

                List<String> l = Collections.unmodifiableList(
                        mode == Mode.INCLUDE_OTHER_COLUMN ?
                                Arrays.asList("0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "other") :
                                Arrays.asList("0", "1", "2", "3", "4", "5", "6", "7", "8", "9"));

                CategoricalMetaData cm = new CategoricalMetaData(outputColumn, l);

                outMeta.add(cm);
            } else {
                outMeta.add(origMeta.get(i));
            }
        }

        return inputSchema.newSchema(outMeta);
    }

    @Override
    public String outputColumnName() {
        return outputColumn;
    }

    @Override
    public String[] outputColumnNames() {
        return new String[]{outputColumn};
    }

    @Override
    public String[] columnNames() {
        return new String[]{inputColumn};
    }

    @Override
    public String columnName() {
        return inputColumn;
    }

    @Override
    public void setInputSchema(Schema schema){
        super.setInputSchema(schema);

        columnIdx = schema.getIndexOfColumn(inputColumn);
        Preconditions.checkState(columnIdx >= 0, "Input column \"%s\" not found in schema", inputColumn);
    }
}
