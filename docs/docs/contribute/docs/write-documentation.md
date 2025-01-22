---
authors: Fragcolor & contributors
license: CC-BY-SA-4.0
---

# Write Documentation

!!! note

    Our documentation follows the [Microsoft Style Guide](https://docs.microsoft.com/en-us/style-guide/welcome/). Do refer to it if you are unsure of the documentation style!

In this chapter, you will be learning how to write documentation for Fragcolor.

Pre-requisites:

- [Ready the Development Environment](../getting-started.md)

- [Build Shards](../../contribute/code/build-shards.md)

- [Install MkDocs](./start-documenting.md)

## Navigating the Docs

Documentation exists under the `/docs` folder of the Shards repository. We will list down notable folders that you will want to take note of.

For Shards API Documentation:

- `/docs/docs/reference`: contains the pages for the "Reference" category of the site.

- `/docs/details`: holds the files containing the details of each shard.

- `/docs/samples`: contains the code and output of the examples for each shard.

Content from the three folders above is combined to make up each API documentation page.

![A shard's API page and their associated folders.](assets/docs-folders.png)

For the other documentation pages:

- `/docs/docs/contribute`: contains the pages for the "Contribute" category of the site.

- `/docs/docs/learn`: contains the pages for the "Learn" category of the site.

## Previewing the Docs

1. Open your MinGW terminal.

2. Navigate to where your docs folder is located using the command `cd $(cygpath -u '(X)')`, where (X) is the directory of your folder.

        cd $(cygpath -u '(X)')

    If your docs folder is located at `C:\Fragcolor\Shards\docs`, the command used in the MinGW terminal would be `cd $(cygpath -u 'C:\Fragcolor\Shards\docs')`.

3. Run the following command to start the server:

        mkdocs serve

4. Enter the following in your browser to preview the site:

        http://127.0.0.1:8000/


## Pages and Levels

The root directory for our site is located at `/docs/docs`, with `index.md` as the file for our homepage. Subsequent directories are placed within folders.

![A level in the doc's hierarchy.](assets/docs-hierarchy.png)

Each level contains:

- An `index.md` file as the landing page

- An (optional) `assets` folder to hold media used in that level

- A `.pages` file to organize the order of our pages

    ![The .pages file helps to structure the hierarchy of each level.](assets/editing-ref-pages.png)

    !!! note
        If `...` is used within `.pages`, files in the same hierarchy, that have not been mentioned in `.pages` will be arranged in alphabetical order.

        ![Using ... to automatically list pages.](assets/editing-ref-pages-dots.png)

## Generating Documentation for the Shards API

You might have observed that the `/docs/docs/reference/shards` folder is conspicuously empty. This is to allow for the faster previewing of your static site without having to load all the pages for the Shards API.

To generate the documentation for the Shards API:

1. Launch your MinGW terminal.

2. Navigate to where your Shards repository is located using the command `cd $(cygpath -u '(X)')`, where (X) is the directory of your folder.

        cd $(cygpath -u '(X)')

    If your repository is located at `C:\Fragcolor\Shards`, the command used in the MinGW terminal would be `cd $(cygpath -u 'C:\Fragcolor\Shards')`.

 3. Input the following command:

        ./build/Debug/shards new ./docs/generate.shs

!!! note
    The **Debug** version of Shards should be used for the generating of the Shards API pages.

Once done, you will notice that the directory is now filled with folders and Markdown files of various shards.

## Generating Log Files for Examples

Did you realize that the "Output" segments for your code examples are empty too? This is due to how a `.log` file is required for each example to show its output.

To generate the `.log` files for your code examples:

1. Launch your MinGW terminal.

2. Navigate to where your Shards repository is located using the command `cd $(cygpath -u '(X)')`, where (X) is the directory of your folder.

        cd $(cygpath -u '(X)')

    If your repository is located at `C:\Fragcolor\Shards`, the command used in the MinGW terminal would be `cd $(cygpath -u 'C:\Fragcolor\Shards')`.

 3. Input the following command:

        ./run-samples.sh

When the script is done, you will notice new `.log` files within your "Samples" folder. These will be shown in your documentation as the "Output" of your code examples.

![The results with the .edn.log file.](assets/editing-examples-log-md.png)

![The log file will populate the "Output" segment.](assets/editing-examples-site-output.png)

## Updating the Shards API Docs

As you have seen from the segment on [navigating the docs](#navigating-the-docs), a shard's API documentation is generated from three different locations:

1. `/docs/details` folder

2. `/docs/samples` folder

3. `/docs/docs/reference` folder

Each segment will require a different approach to making changes. The following are the steps required to work with each segment:

### Details

1. Navigate to the `/docs/details` folder.

2. Search for the `.md` file with the shard's name. We will be using the `Msg` shard in this example.

    !!! note
        If the file does not exist, create a new `.md` file with the shard's name (e.g. `Msg.md`).

    ![Finding the .md file for a shard's details.](assets/editing-details-directory.png)

3. Amend the text within the `.md` file.

    ![The details are in the .md file.](assets/editing-details-md.png)

4. The changes will be reflected on the site.

    ![The details are on the site.](assets/editing-details-site.png)

### Examples

1. Navigate to the `/docs/samples` folder and locate the folder with the shard's name.

2. The examples are `.edn` files named in numerical order, starting from 1.

    ![Finding the .edn files for a shard's examples.](assets/editing-examples-directory.png)

3. Amend the code within the `.edn` file, or add a new `.edn` file to create a new example.

    ![The code within the .edn file.](assets/editing-examples-md.png)

4. Amend the code within the `.edn.log` file of the same number to reflect the results of the example.

    ![The results with the .edn.log file.](assets/editing-examples-log-md.png)

5. The changes to the `.edn` file will be reflected in the "Code" portion of the "Examples" segment on the site.

    ![The code is reflected on the site.](assets/editing-examples-site-code.png)

6. The changes to the `.edn.log` file will be reflected in the "Output" portion of the "Examples" segment on the site.

    ![The results are reflected on the site.](assets/editing-examples-site-output.png)


### Reference

Since the files within the `/docs/docs/reference` folder were generated by running a script, the editing of the description text in it will require additional steps.

In this segment, we will learn how to edit the following:

- The basic description of a shard

- The description text for the input and output

- The description text for the shard's parameters

![A breakdown of the description texts being covered.](assets/docs-reference-breakdown.png)

Editing these description texts will require looking at the source .cpp and .hpp files.

#### Basic Description

Firstly, locate the struct of the shard you wish to amend. For this tutorial, we will be using the `String.Join` shard as an example.

1. Navigate to the `/src/core/shards` folder and select the file that most likely contains the target shard. For this example, we will select the `strings.cpp` file.

    ![The .cpp file exists in the "src/core/shards" folder.](assets/editing-ref-directory.png)

2. Search for where the shards are being registered and check the struct of the shard. In this example, `String.Join` has a struct named `Join`.

    ![Check for the struct being registered to the shard.](assets/editing-ref-cpp-registershard.png)

    !!! note
        There might be instances whereby the struct is named differently from the shard. It is therefore important to check the struct's name first before proceeding.

3. Navigate to the struct in the same file to access the `help()` function. The basic description of the shard is located in the return statement and can be edited from there.

    ![The basic description of the shard is located in the `help()` function.](assets/editing-re)

#### Input and Output

Within the same struct, you can find the description text for the input and output within the `inputHelp()` and `outputHelp()` functions.

![The descriptions of the input and output are located in the `inputHelp()` and `outputHelp()` functions.](assets/editing-ref-cpp-struct-input-output.png)

#### Parameters

For the description of the parameters, look for the `parameters()` function within the struct.

You will see either:

- The parameter description within the function,

- OR a variable being returned.

In the case of `String.Join`, a variable `params` is returned. Search for the variable used to find the description text of the parameters.

![The descriptions of the parameters are placed within a params variable.](assets/editing-ref-cpp-struct-parameters.png)

![The descriptions of the parameters are found within the params variable.](assets//editing-ref-cpp-struct-params.png)

## Formatting the Docs

Our documentation is formatted with Markdown. Do check out the [cheatsheet](https://www.markdownguide.org/cheat-sheet/) if you are new to Markdown or need a refresher.

Material for MkDocs allows you to add fancier stuff to your documentation, such as admonitions and annotated code blocks. For more information on what you can achieve, do check out the reference page [here](https://squidfunk.github.io/mkdocs-material/reference/).

!!! note

    This is an admonition created with Material for MkDocs!

Great job reaching the end of the tutorial! We hope this has been useful in helping you start your documentation writing endeavors.

## Overview ##

1. `/docs/docs/contribute`: "Contribute" pages

2. `/docs/docs/learn`: "Learn" pages

3. `/docs/docs/reference`: API "Reference" pages

4. `/docs/details`: API Details

5. `/docs/samples`: API Examples

6. `/src/core/shards`: API Basic/Input/Output/Parameters Description

--8<-- "includes/license.md"
