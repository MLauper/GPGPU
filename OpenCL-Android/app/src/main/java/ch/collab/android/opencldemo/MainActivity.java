package ch.collab.android.opencldemo;

import android.graphics.Color;
import android.os.Bundle;
import android.view.Gravity;
import android.view.ViewGroup;
import android.widget.TableLayout;
import android.widget.TableRow;
import android.widget.TextView;
import android.support.design.widget.FloatingActionButton;
import android.support.design.widget.Snackbar;
import android.support.v7.app.AppCompatActivity;
import android.support.v7.widget.Toolbar;
import android.view.View;
import android.view.Menu;
import android.view.MenuItem;

public class MainActivity extends AppCompatActivity {

    // Used to load the 'native-lib' library on application startup.
    static {
        System.loadLibrary("native-lib");
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        Toolbar toolbar = (Toolbar) findViewById(R.id.toolbar);
        setSupportActionBar(toolbar);

        FloatingActionButton fab = (FloatingActionButton) findViewById(R.id.fab);
        fab.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                Snackbar.make(view, "Reloading OpenCL Demo...", Snackbar.LENGTH_LONG)
                        .setAction("Action", null).show();
                openCLDemo();
            }
        });

        //// Example of a call to a native method
        //TextView tv = (TextView) findViewById(R.id.sample_text);
        //tv.setText(stringFromJNI());

        openCLDemo();
    }

    /**
     * This function will invoke JNI functions and print the output on the screen.
     */
    private void openCLDemo() {

        // Get table for output
        TableLayout tableLayout = (TableLayout) findViewById(R.id.table_main);

        addEntry(tableLayout, "Num of Platforms", getNumOfPlatforms());


    }

    private void addEntry(TableLayout tableLayout, String property, String value){

        // Create new Row to be added
        TableRow tableRow = new TableRow(this);
        tableRow.setWeightSum(2);
        //tableRow.setLayoutParams(new TableRow.LayoutParams(TableRow.LayoutParams.MATCH_PARENT, TableRow.LayoutParams.MATCH_PARENT));

        // Add Property column
        TextView propertyTextView = new TextView(this);
        propertyTextView.setText(property);
        propertyTextView.setTextColor(Color.WHITE);
        propertyTextView.setGravity(Gravity.LEFT);
        propertyTextView.setLayoutParams(new TableRow.LayoutParams(TableRow.LayoutParams.WRAP_CONTENT, TableRow.LayoutParams.WRAP_CONTENT, 1.0f));
        propertyTextView.setPadding(convertDPtoPixel(5),0,0,0);
        tableRow.addView(propertyTextView);

        // Add Value column
        TextView valueTextView = new TextView(this);
        valueTextView.setText(value);
        valueTextView.setTextColor(Color.WHITE);
        valueTextView.setGravity(Gravity.LEFT);
        valueTextView.setLayoutParams(new TableRow.LayoutParams(TableRow.LayoutParams.WRAP_CONTENT, TableRow.LayoutParams.WRAP_CONTENT, 1.0f));
        valueTextView.setPadding(convertDPtoPixel(5),0,0,0);
        tableRow.addView(valueTextView);

        // Add row to table
        tableLayout.addView(tableRow);
    }

    private int convertDPtoPixel(int dp) {
        final float scale = getResources().getDisplayMetrics().density;
        return (int) (dp * scale + 0.5f);
    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        // Inflate the menu; this adds items to the action bar if it is present.
        getMenuInflater().inflate(R.menu.menu_main, menu);
        return true;
    }

    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        // Handle action bar item clicks here. The action bar will
        // automatically handle clicks on the Home/Up button, so long
        // as you specify a parent activity in AndroidManifest.xml.
        int id = item.getItemId();

        //noinspection SimplifiableIfStatement
        if (id == R.id.action_settings) {
            return true;
        }

        return super.onOptionsItemSelected(item);
    }

    /**
     * A native method that is implemented by the 'native-lib' native library,
     * which is packaged with this application.
     */
    public native String stringFromJNI();
    public native String getNumOfPlatforms();
}
